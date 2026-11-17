import type { Character, OfflineVideoTTSConfig } from '../types'

export function hasExplicitOfflineVideoTTS(preference?: OfflineVideoTTSConfig | null): boolean {
  if (!preference) return false
  return !!(preference.provider?.trim() || preference.model?.trim() || preference.voice?.trim())
}

/** Defaults from the character card (live voice), not the global component catalog default. */
export function characterCardOfflineTTS(character: Character): OfflineVideoTTSConfig {
  const provider = (character.voice_provider || character.components?.tts || '').trim() || 'qwen'
  const model = (character.components?.tts_model || '').trim()
  const voice = (character.voice_type || '').trim()
  return { provider, model, voice }
}