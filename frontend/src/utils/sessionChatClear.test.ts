import { describe, expect, it } from 'vitest'
import {
  characterChatHistoryClearedAtMs,
  emptyLocalChatMessages,
  filterHistoryAfterChatClear,
} from './sessionChatClear'
import type { ChatMessage } from '../composables/useChat'

describe('sessionChatClear', () => {
  it('emptyLocalChatMessages returns an empty array', () => {
    expect(emptyLocalChatMessages()).toEqual([])
  })

  it('filterHistoryAfterChatClear drops live-session history before cutoff', () => {
    const clearedAt = 1_000_000
    const liveId = 'sess-live'
    const messages: ChatMessage[] = [
      { role: 'user', content: 'old', timestamp: clearedAt - 1000, isHistory: true, sessionId: liveId },
      { role: 'assistant', content: 'kept', timestamp: clearedAt + 1, isHistory: true, sessionId: liveId },
      { role: 'user', content: 'other session', timestamp: 0, isHistory: true, sessionId: 'other' },
    ]
    const filtered = filterHistoryAfterChatClear(messages, liveId, clearedAt)
    expect(filtered).toHaveLength(2)
    expect(filtered.map((m) => m.content)).toEqual(['kept', 'other session'])
  })

  it('character cutoff hides prior sessions after relaunch with new session id', () => {
    const charCutoff = 2_000_000
    const messages: ChatMessage[] = [
      { role: 'user', content: 'stale', timestamp: charCutoff - 1, isHistory: true, sessionId: 'old-session' },
      { role: 'user', content: 'fresh', timestamp: charCutoff + 1, isHistory: true, sessionId: 'new-session' },
    ]
    const filtered = filterHistoryAfterChatClear(messages, 'new-session', 0, charCutoff)
    expect(filtered.map((m) => m.content)).toEqual(['fresh'])
  })

  it('characterChatHistoryClearedAtMs parses RFC3339', () => {
    const iso = '2026-11-15T10:00:00.123Z'
    expect(characterChatHistoryClearedAtMs(iso)).toBe(Date.parse(iso))
    expect(characterChatHistoryClearedAtMs('')).toBe(0)
  })
})