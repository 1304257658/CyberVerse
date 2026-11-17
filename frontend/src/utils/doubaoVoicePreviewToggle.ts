/** Whether the next preview click should stop (cancel) instead of starting playback. */
export function shouldCancelDoubaoVoicePreview(
  activeVoice: string | null,
  requestedVoice: string,
): boolean {
  const trimmed = requestedVoice.trim()
  if (!trimmed) return false
  return activeVoice === trimmed
}