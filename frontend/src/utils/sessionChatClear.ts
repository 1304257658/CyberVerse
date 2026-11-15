import type { ChatMessage } from '../composables/useChat'

/** In-memory UI state after a successful clear-chat response. */
export function emptyLocalChatMessages(): ChatMessage[] {
  return []
}

/**
 * Filters paginated history so messages from the live session before a cutoff
 * do not reappear after refresh (belt-and-suspenders if session.json was truncated).
 */
export function filterHistoryAfterChatClear(
  historyMessages: ChatMessage[],
  liveSessionId: string,
  clearedAtMs: number,
): ChatMessage[] {
  if (!liveSessionId || clearedAtMs <= 0) {
    return historyMessages
  }
  return historyMessages.filter((message) => {
    if (!message.isHistory) {
      return true
    }
    if (message.sessionId && message.sessionId !== liveSessionId) {
      return true
    }
    if (!message.sessionId) {
      return true
    }
    return message.timestamp >= clearedAtMs
  })
}