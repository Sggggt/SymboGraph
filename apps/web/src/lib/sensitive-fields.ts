export const SENSITIVE_FIELD_KEY_PROTOCOL_VERSION =
  "semantic_sensitive_field_key_segments_v1" as const;

export type SensitiveFieldKeyReason =
  | "private_profile_prompt_container"
  | "private_profile_document"
  | "api_key"
  | "provider_response"
  | "system_prompt"
  | "credential"
  | "authorization"
  | "secret"
  | "token"
  | "dangerous_compound";

export type PESensitiveKeyKind =
  | "provider_raw_response"
  | "credentials";

const singularSegments = new Map([
  ["credentials", "credential"],
  ["keys", "key"],
  ["prompts", "prompt"],
  ["responses", "response"],
  ["secrets", "secret"],
  ["tokens", "token"],
]);

const safeStatusTails = new Set([
  "available",
  "configured",
  "count",
  "counts",
  "digest",
  "enabled",
  "exposed",
  "fields",
  "hash",
  "present",
  "protocol",
  "redacted",
  "redaction",
  "required",
  "source",
  "status",
  "version",
]);

const safeTokenOperationalSegments = new Set([
  "audit",
  "batch",
  "budget",
  "cache",
  "chunk",
  "client",
  "consumed",
  "consumption",
  "concept",
  "content",
  "context",
  "conversation",
  "cost",
  "count",
  "counts",
  "end",
  "estimate",
  "estimated",
  "extraction",
  "fixed",
  "history",
  "hint",
  "input",
  "label",
  "length",
  "limit",
  "local",
  "max",
  "maximum",
  "mid",
  "min",
  "minimum",
  "model",
  "neighbor",
  "original",
  "output",
  "overlap",
  "package",
  "per",
  "protocol",
  "quality",
  "question",
  "remaining",
  "request",
  "selected",
  "size",
  "span",
  "start",
  "sufficiency",
  "task",
  "tokenizer",
  "total",
  "usage",
  "used",
  "version",
  "window",
  "write",
]);

const dangerousStorageSegments = new Set([
  "archive",
  "backup",
  "blob",
  "bundle",
  "copy",
  "snapshot",
  "value",
]);

const publicPrivateCompactKeys = new Set(["promptpack"]);

const compactDangerousPatterns = [
  /^.*apikey(?:backup|blob|bundle|copy|secret|value)?$/,
  /^.*(?:access|admin|api|auth|bearer|client|id|oauth|refresh|session|signing)?token(?:archive|backup|blob|bundle|copy|key|secret|value)?$/,
  /^.*credential(?:archive|backup|blob|bundle|copy|s)?$/,
  /^.*(?:providerrawresponse|rawproviderresponse|providerresponse|providerpayload|rawproviderpayload)(?:archive|backup|blob|bundle|copy|snapshot|value)?$/,
  /^.*system(?:content|prompt)(?:archive|backup|blob|bundle|copy|snapshot|value)?$/,
];

function pythonCasefoldForAsciiSegments(value: string): string {
  // These are the only NFKC-stable Unicode case-fold expansions that change
  // the ASCII alphanumeric segments produced by Python's str.casefold().
  return value
    .replace(/[\u00df\u1e9e]/g, "ss")
    .replace(/\u01f0/g, "j\u030c")
    .replace(/\u1e96/g, "h\u0331")
    .replace(/\u1e97/g, "t\u0308")
    .replace(/\u1e98/g, "w\u030a")
    .replace(/\u1e99/g, "y\u030a")
    .toLowerCase();
}

export function semanticSensitiveKeySegments(value: unknown): string[] {
  const camelSegmented = String(value)
    .trim()
    .normalize("NFKC")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1_$2")
    .replace(/([a-z0-9])([A-Z])/g, "$1_$2");
  const rawSegments =
    pythonCasefoldForAsciiSegments(camelSegmented).match(/[a-z0-9]+/g) ?? [];
  return rawSegments.map(
    (segment) => singularSegments.get(segment) ?? segment,
  );
}

function hasContiguousSegments(
  segments: string[],
  expected: string[],
): boolean {
  for (
    let index = 0;
    index <= segments.length - expected.length;
    index += 1
  ) {
    if (
      expected.every(
        (segment, offset) => segments[index + offset] === segment,
      )
    ) {
      return true;
    }
  }
  return false;
}

function safeObservabilityKey(segments: string[]): boolean {
  if (segments.length === 0) return false;
  if (segments[0] === "has") return true;
  if (
    segments.some((segment) =>
      ["expose", "exposed", "exposes"].includes(segment),
    )
  ) {
    return true;
  }
  return safeStatusTails.has(segments[segments.length - 1]);
}

function safeTokenOperationalKey(segments: string[]): boolean {
  if (!segments.includes("token") || segments.length <= 1) return false;
  const nonToken = segments.filter((segment) => segment !== "token");
  return (
    nonToken.length > 0 &&
    nonToken.every((segment) => safeTokenOperationalSegments.has(segment))
  );
}

export function sensitiveFieldKeyReason(
  value: unknown,
  options: { includePublicPrivate?: boolean } = {},
): SensitiveFieldKeyReason | null {
  const segments = semanticSensitiveKeySegments(value);
  if (segments.length === 0) return null;
  const compact = segments.join("");
  if (
    options.includePublicPrivate &&
    publicPrivateCompactKeys.has(compact)
  ) {
    return "private_profile_prompt_container";
  }

  const hasProfileDocument =
    hasContiguousSegments(segments, ["profile", "json"]) ||
    compact.startsWith("profilejson");
  const hasApiKey =
    hasContiguousSegments(segments, ["api", "key"]) ||
    compact.includes("apikey");
  const hasProviderResponse =
    (segments.includes("provider") &&
      (segments.includes("response") || segments.includes("payload"))) ||
    compact.includes("providerrawresponse") ||
    compact.includes("rawproviderresponse") ||
    compact.includes("providerresponse") ||
    compact.includes("providerpayload") ||
    compact.includes("rawproviderpayload");
  const hasSystemPrompt =
    (segments.includes("system") &&
      (segments.includes("prompt") || segments.includes("content"))) ||
    compact.includes("systemprompt") ||
    compact.includes("systemcontent");
  const hasCredential =
    segments.includes("credential") || compact.includes("credential");
  const hasAuthorization =
    segments.includes("auth") ||
    segments.includes("authorization") ||
    ["auth", "authorization", "authorizationheader"].includes(compact) ||
    compact.startsWith("authtoken");
  const hasSecret =
    segments.includes("password") ||
    segments.includes("secret") ||
    compact.endsWith("password") ||
    compact.endsWith("secret");
  const hasToken =
    segments.includes("token") || compactDangerousPatterns[1].test(compact);
  const hasDangerousStorageSemantics = segments.some((segment) =>
    dangerousStorageSegments.has(segment),
  );

  if (
    (hasApiKey ||
      hasProviderResponse ||
      hasCredential ||
      hasAuthorization ||
      hasSecret ||
      hasSystemPrompt) &&
    !hasDangerousStorageSemantics &&
    safeObservabilityKey(segments)
  ) {
    return null;
  }
  if (hasToken && safeTokenOperationalKey(segments)) return null;

  if (hasProfileDocument) return "private_profile_document";
  if (hasApiKey) return "api_key";
  if (hasProviderResponse) return "provider_response";
  if (hasSystemPrompt) return "system_prompt";
  if (hasCredential) return "credential";
  if (hasAuthorization) return "authorization";
  if (hasSecret) return "secret";
  if (hasToken) return "token";
  if (compactDangerousPatterns.some((pattern) => pattern.test(compact))) {
    return "dangerous_compound";
  }
  return null;
}

export function peSensitiveKeyKind(
  value: unknown,
): PESensitiveKeyKind | null {
  const segments = semanticSensitiveKeySegments(value);
  const reason = sensitiveFieldKeyReason(value, {
    includePublicPrivate: true,
  });
  if (reason === "provider_response") return "provider_raw_response";
  if (reason !== null) return "credentials";

  const dangerousStorage = segments.some((segment) =>
    dangerousStorageSegments.has(segment),
  );
  for (const [left, right] of [
    ["raw", "output"],
    ["raw", "response"],
  ]) {
    for (let index = 0; index < segments.length - 1; index += 1) {
      if (segments[index] !== left || segments[index + 1] !== right) {
        continue;
      }
      if (index + 2 === segments.length || dangerousStorage) {
        return "provider_raw_response";
      }
    }
  }

  const lastHeaderIndex = segments.reduce(
    (lastIndex, segment, index) =>
      ["header", "headers"].includes(segment) ? index : lastIndex,
    -1,
  );
  if (
    lastHeaderIndex >= 0 &&
    (lastHeaderIndex === segments.length - 1 || dangerousStorage)
  ) {
    return "credentials";
  }
  return null;
}
