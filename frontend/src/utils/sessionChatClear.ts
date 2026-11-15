import type { ChatMessage } from '../composables/useChat'

/** In-memory UI state after a successful clear-chat response. */
export function emptyLocalChatMessages(): ChatMessage[] {
  return []
}

export function characterChatHistoryClearedAtMs(clearedAt?: string): number {
  if (!clearedAt?.trim()) return 0
  const ms = Date.parse(clearedAt)
  return Number.isFinite(ms) ? ms : 0
}

/**
 * Filters paginated history so cleared conversation does not reappear after reload or relaunch.
 * Character-level cutoff hides all sessions before clear; live-session cutoff covers in-memory belt-and-suspenders.
 */
export function filterHistoryAfterChatClear(
  historyMessages: ChatMessage[],
  liveSessionId: string,
  liveClearedAtMs: number,
  characterClearedAtMs = 0,
): ChatMessage[] {
  const charCutoff = characterClearedAtMs > 0 ? characterClearedAtMs : 0
  const liveCutoff = liveSessionId && liveClearedAtMs > 0 ? liveClearedAtMs : 0
  if (charCutoff <= 0 && liveCutoff <= 0) {
    return historyMessages
  }
  return historyMessages.filter((message) => {
    if (!message.isHistory) {
      return true
    }
    if (charCutoff > 0 && message.timestamp < charCutoff) {
      return false
    }
    if (liveCutoff > 0 && message.sessionId === liveSessionId && message.timestamp < liveCutoff) {
      return false
    }
    return true
  })
}