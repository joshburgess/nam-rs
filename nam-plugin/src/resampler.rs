use nam_core::{Dsp, Sample};
use rubato::{FftFixedInOut, Resampler};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum AudioProcessError {
    None = 0,
    InputCapacity = 1,
    ModelCapacity = 2,
    OutputCapacity = 3,
    ToModel = 4,
    ToHost = 5,
}

impl AudioProcessError {
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::None => "",
            Self::InputCapacity => "Host input exceeded the preallocated resampler capacity",
            Self::ModelCapacity => "Model-rate audio exceeded the preallocated buffer capacity",
            Self::OutputCapacity => "Host output exceeded the preallocated resampler capacity",
            Self::ToModel => "Host-to-model resampling failed",
            Self::ToHost => "Model-to-host resampling failed",
        }
    }

    pub(super) fn from_raw(value: u8) -> Self {
        match value {
            1 => Self::InputCapacity,
            2 => Self::ModelCapacity,
            3 => Self::OutputCapacity,
            4 => Self::ToModel,
            5 => Self::ToHost,
            _ => Self::None,
        }
    }
}

pub(super) struct ResamplerState {
    to_model: FftFixedInOut<f64>,
    to_host: FftFixedInOut<f64>,
    pub(super) input_pending: VecDeque<f64>,
    pub(super) model_rate_pending: VecDeque<f64>,
    pub(super) output_pending: VecDeque<f64>,
    to_model_chunk: usize,
    to_host_chunk: usize,
    pub(super) model_input: Vec<Sample>,
    pub(super) model_output: Vec<Sample>,
    to_model_input: Vec<Vec<f64>>,
    to_model_output: Vec<Vec<f64>>,
    to_host_input: Vec<Vec<f64>>,
    to_host_output: Vec<Vec<f64>>,
}

impl ResamplerState {
    pub(super) fn new(
        host_rate: usize,
        model_rate: usize,
        max_buffer_size: usize,
    ) -> Result<Self, rubato::ResamplerConstructionError> {
        let to_model = FftFixedInOut::<f64>::new(host_rate, model_rate, 128, 1)?;
        let to_host = FftFixedInOut::<f64>::new(model_rate, host_rate, 128, 1)?;

        let to_model_chunk = to_model.input_frames_max();
        let to_model_output_frames = to_model.output_frames_max();
        let to_host_chunk = to_host.input_frames_max();
        let to_host_output_frames = to_host.output_frames_max();
        let max_model_buf = max_buffer_size
            .div_ceil(to_model_chunk)
            .saturating_mul(to_model_output_frames);
        let pending_capacity = max_buffer_size
            .saturating_add(to_model_chunk)
            .saturating_add(to_host_output_frames);

        Ok(Self {
            to_model,
            to_host,
            input_pending: VecDeque::with_capacity(pending_capacity),
            model_rate_pending: VecDeque::with_capacity(
                max_model_buf.saturating_add(to_host_chunk),
            ),
            output_pending: VecDeque::with_capacity(
                max_buffer_size.saturating_add(to_host_output_frames),
            ),
            to_model_chunk,
            to_host_chunk,
            model_input: vec![0.0; max_model_buf],
            model_output: vec![0.0; max_model_buf],
            to_model_input: vec![vec![0.0; to_model_chunk]; 1],
            to_model_output: vec![vec![0.0; to_model_output_frames]; 1],
            to_host_input: vec![vec![0.0; to_host_chunk]; 1],
            to_host_output: vec![vec![0.0; to_host_output_frames]; 1],
        })
    }

    pub(super) fn reset(&mut self) {
        self.input_pending.clear();
        self.model_rate_pending.clear();
        self.output_pending.clear();
    }

    pub(super) fn process(
        &mut self,
        model: &mut dyn Dsp,
        input: &[Sample],
        output: &mut [Sample],
    ) -> Result<(), AudioProcessError> {
        let num_samples = input.len();
        if self.input_pending.len().saturating_add(num_samples) > self.input_pending.capacity() {
            return Err(AudioProcessError::InputCapacity);
        }

        for &sample in input {
            self.input_pending
                .push_back(nam_core::dsp::sample_to_f64(sample));
        }

        while self.input_pending.len() >= self.to_model_chunk {
            for sample in &mut self.to_model_input[0][..self.to_model_chunk] {
                *sample = self.input_pending.pop_front().unwrap_or(0.0);
            }
            let (_, output_frames) = self
                .to_model
                .process_into_buffer(&self.to_model_input, &mut self.to_model_output, None)
                .map_err(|_| AudioProcessError::ToModel)?;
            if self.model_rate_pending.len().saturating_add(output_frames)
                > self.model_rate_pending.capacity()
            {
                return Err(AudioProcessError::ModelCapacity);
            }
            for &sample in &self.to_model_output[0][..output_frames] {
                self.model_rate_pending.push_back(sample);
            }
        }

        let model_samples = self.model_rate_pending.len();
        if model_samples > 0 {
            if model_samples > self.model_input.len() {
                return Err(AudioProcessError::ModelCapacity);
            }
            for sample in &mut self.model_input[..model_samples] {
                *sample = nam_core::dsp::sample_from_f64(
                    self.model_rate_pending.pop_front().unwrap_or(0.0),
                );
            }

            model.process(
                &self.model_input[..model_samples],
                &mut self.model_output[..model_samples],
            );

            if model_samples > self.model_rate_pending.capacity() {
                return Err(AudioProcessError::ModelCapacity);
            }
            for &sample in &self.model_output[..model_samples] {
                self.model_rate_pending
                    .push_back(nam_core::dsp::sample_to_f64(sample));
            }
        }

        while self.model_rate_pending.len() >= self.to_host_chunk {
            for sample in &mut self.to_host_input[0][..self.to_host_chunk] {
                *sample = self.model_rate_pending.pop_front().unwrap_or(0.0);
            }
            let (_, output_frames) = self
                .to_host
                .process_into_buffer(&self.to_host_input, &mut self.to_host_output, None)
                .map_err(|_| AudioProcessError::ToHost)?;
            if self.output_pending.len().saturating_add(output_frames)
                > self.output_pending.capacity()
            {
                return Err(AudioProcessError::OutputCapacity);
            }
            for &sample in &self.to_host_output[0][..output_frames] {
                self.output_pending.push_back(sample);
            }
        }

        for sample in output.iter_mut().take(num_samples) {
            *sample =
                nam_core::dsp::sample_from_f64(self.output_pending.pop_front().unwrap_or(0.0));
        }
        Ok(())
    }
}
