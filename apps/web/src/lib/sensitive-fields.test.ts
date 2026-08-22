import { describe, expect, it } from "vitest";

import {
  SENSITIVE_FIELD_KEY_PROTOCOL_VERSION,
  peSensitiveKeyKind,
  sensitiveFieldKeyReason,
} from "./sensitive-fields";


describe("semantic sensitive-field classification", () => {
  it("keeps the public protocol stable", () => {
    expect(SENSITIVE_FIELD_KEY_PROTOCOL_VERSION).toBe(
      "semantic_sensitive_field_key_segments_v1",
    );
  });

  it.each([
    ["api_key", "api_key", "credentials"],
    ["authorization_header", "authorization", "credentials"],
    ["provider_raw_response", "provider_response", "provider_raw_response"],
    ["system_prompt", "system_prompt", "credentials"],
    ["context_token_budget", null, null],
    ["has_api_key", null, null],
  ] as const)("classifies %s", (key, reason, peKind) => {
    expect(sensitiveFieldKeyReason(key, { includePublicPrivate: true })).toBe(
      reason,
    );
    expect(peSensitiveKeyKind(key)).toBe(peKind);
  });
});
