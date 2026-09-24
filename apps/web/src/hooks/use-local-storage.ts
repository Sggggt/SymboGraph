import { useCallback, useRef, useSyncExternalStore } from "react";

const LOCAL_STORAGE_EVENT = "codex-local-storage";
const snapshotCache = new Map<string, { raw: string | null; parsed: unknown }>();
const persistenceWarningKeys = new Set<string>();

const LEGACY_QA_PAYLOAD_PREFIXES = [
  "qa.turns.",
  "qa.draftAnswer.",
  "qa.citations.",
  "qa.trace.",
  "qa.latestRun.",
  "qa.conversationState.",
] as const;

export function isLegacyQaPayloadStorageKey(key: string): boolean {
  return LEGACY_QA_PAYLOAD_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function storageQuotaExceeded(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const candidate = error as { code?: unknown; name?: unknown };
  return candidate.name === "QuotaExceededError"
    || candidate.name === "NS_ERROR_DOM_QUOTA_REACHED"
    || candidate.code === 22
    || candidate.code === 1014;
}

export function cleanupLegacyQaPayloadStorage(storage?: Storage): string[] {
  const target = storage ?? (typeof window === "undefined" ? null : window.localStorage);
  if (!target) {
    return [];
  }

  const candidates: Array<{ key: string; size: number }> = [];
  try {
    for (let index = 0; index < target.length; index += 1) {
      const key = target.key(index);
      if (!key || !isLegacyQaPayloadStorageKey(key)) {
        continue;
      }
      const value = target.getItem(key);
      candidates.push({ key, size: key.length + (value?.length ?? 0) });
    }
  } catch (error) {
    console.warn("Error inspecting legacy QA localStorage payloads:", error);
    return [];
  }

  candidates.sort((left, right) => right.size - left.size || left.key.localeCompare(right.key));
  const removed: string[] = [];
  for (const candidate of candidates) {
    try {
      target.removeItem(candidate.key);
      snapshotCache.delete(candidate.key);
      removed.push(candidate.key);
    } catch (error) {
      console.warn(`Error removing legacy localStorage key "${candidate.key}":`, error);
    }
  }
  return removed;
}

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
      snapshotCache.delete(key);
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
    let valueToStore: T;
    try {
      const currentValue = readLocalStorageValue(key, initialValueRef.current);
      valueToStore = value instanceof Function ? value(currentValue) : value;
    } catch (error) {
      console.warn(`Error preparing localStorage key "${key}":`, error);
      return;
    }

    if (typeof window === "undefined") {
      return;
    }

    const serialized = JSON.stringify(valueToStore);
    let persistenceError: unknown = null;
    try {
      window.localStorage.setItem(key, serialized);
    } catch (error) {
      persistenceError = error;
      if (storageQuotaExceeded(error)) {
        cleanupLegacyQaPayloadStorage(window.localStorage);
        try {
          window.localStorage.setItem(key, serialized);
          persistenceError = null;
        } catch (retryError) {
          persistenceError = retryError;
        }
      }
    }

    if (persistenceError === null) {
      persistenceWarningKeys.delete(key);
      snapshotCache.set(key, { raw: serialized, parsed: valueToStore });
    } else {
      let currentRaw: string | null = null;
      try {
        currentRaw = window.localStorage.getItem(key);
      } catch {
        // The in-memory snapshot below is still authoritative for this page.
      }
      snapshotCache.set(key, { raw: currentRaw, parsed: valueToStore });
      if (!persistenceWarningKeys.has(key)) {
        persistenceWarningKeys.add(key);
        console.warn(
          `Error setting localStorage key "${key}"; continuing with in-memory state:`,
          persistenceError,
        );
      }
    }

    emitLocalStorageChange(key);
  }, [key]);

  return [storedValue, setValue] as const;
}
