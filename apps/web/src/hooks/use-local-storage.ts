import { useCallback, useRef, useSyncExternalStore } from "react";

const LOCAL_STORAGE_EVENT = "codex-local-storage";
const snapshotCache = new Map<string, { raw: string | null; parsed: unknown }>();

function emitLocalStorageChange(key: string) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new CustomEvent(LOCAL_STORAGE_EVENT, { detail: { key } }));
}

function subscribeToKey(key: string, onStoreChange: () => void) {
  if (typeof window === "undefined") {
    return () => {};
  }

  const handleStorage = (event: StorageEvent) => {
    if (event.key === key) {
      onStoreChange();
    }
  };

  const handleCustomEvent = (event: Event) => {
    const customEvent = event as CustomEvent<{ key?: string }>;
    if (customEvent.detail?.key === key) {
      onStoreChange();
    }
  };

  window.addEventListener("storage", handleStorage);
  window.addEventListener(LOCAL_STORAGE_EVENT, handleCustomEvent as EventListener);

  return () => {
    window.removeEventListener("storage", handleStorage);
    window.removeEventListener(LOCAL_STORAGE_EVENT, handleCustomEvent as EventListener);
  };
}

function readLocalStorageValue<T>(key: string, initialValue: T): T {
  if (typeof window === "undefined") {
    return initialValue;
  }

  try {
    const item = window.localStorage.getItem(key);
    const cached = snapshotCache.get(key);

    if (cached && cached.raw === item) {
      return cached.parsed as T;
    }

    if (item !== null) {
      const parsed = JSON.parse(item) as T;
      snapshotCache.set(key, { raw: item, parsed });
      return parsed;
    }
  } catch (error) {
    console.warn(`Error reading localStorage key "${key}":`, error);
  }

  snapshotCache.set(key, { raw: null, parsed: initialValue });
  return initialValue;
}

export function useLocalStorage<T>(key: string, initialValue: T) {
  const initialValueRef = useRef(initialValue);
  const storedValue = useSyncExternalStore(
    (onStoreChange) => subscribeToKey(key, onStoreChange),
    () => readLocalStorageValue(key, initialValue),
    () => initialValue,
  );

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const currentValue = readLocalStorageValue(key, initialValueRef.current);
      const valueToStore = value instanceof Function ? value(currentValue) : value;
      if (typeof window !== "undefined") {
        const serialized = JSON.stringify(valueToStore);
        window.localStorage.setItem(key, serialized);
        snapshotCache.set(key, { raw: serialized, parsed: valueToStore });
        emitLocalStorageChange(key);
      }
    } catch (error) {
      console.warn(`Error setting localStorage key "${key}":`, error);
    }
  }, [key]);

  return [storedValue, setValue] as const;
}
