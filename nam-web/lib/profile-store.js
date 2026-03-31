// profile-store.js — IndexedDB persistence for .nam profiles

const DB_NAME = 'nam-web';
const DB_VERSION = 1;
const STORE_NAME = 'profiles';

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(mode, fn) {
  return openDB().then((db) => {
    return new Promise((resolve, reject) => {
      const t = db.transaction(STORE_NAME, mode);
      const store = t.objectStore(STORE_NAME);
      const result = fn(store);
      t.oncomplete = () => resolve(result._result);
      t.onerror = () => reject(t.error);

      // For get/getAll requests, capture the result
      if (result instanceof IDBRequest) {
        result.onsuccess = () => { result._result = result.result; };
      }
    });
  });
}

export async function saveProfile(name, arrayBuffer) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE_NAME, 'readwrite');
    const store = t.objectStore(STORE_NAME);
    const record = {
      name,
      data: arrayBuffer,
      size: arrayBuffer.byteLength,
      createdAt: Date.now(),
    };
    const req = store.add(record);
    req.onsuccess = () => resolve(req.result);
    t.onerror = () => reject(t.error);
  });
}

export async function listProfiles() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE_NAME, 'readonly');
    const store = t.objectStore(STORE_NAME);
    const req = store.getAll();
    req.onsuccess = () => {
      resolve(
        req.result.map(({ id, name, size, createdAt }) => ({
          id,
          name,
          size,
          createdAt,
        }))
      );
    };
    t.onerror = () => reject(t.error);
  });
}

export async function loadProfile(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE_NAME, 'readonly');
    const store = t.objectStore(STORE_NAME);
    const req = store.get(id);
    req.onsuccess = () => {
      if (req.result) {
        resolve(req.result.data);
      } else {
        reject(new Error(`Profile ${id} not found`));
      }
    };
    t.onerror = () => reject(t.error);
  });
}

export async function deleteProfile(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE_NAME, 'readwrite');
    const store = t.objectStore(STORE_NAME);
    store.delete(id);
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}
