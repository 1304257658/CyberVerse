import type { ComponentOption, ComponentsResponse } from '../types'
import {
  LOCAL_COSYVOICE3_DEFAULT_VOICE,
  isCosyVoiceTTSModel,
  localCosyVoice3VoiceLabel,
} from './voice'

export const LOCAL_COSYVOICE3_PROVIDER = 'cosyvoice3'
export { LOCAL_COSYVOICE3_DEFAULT_VOICE, localCosyVoice3VoiceLabel }

export type TTSProviderKind = 'local_cosyvoice3' | 'cloud' | string

export function isLocalCosyVoice3Provider(provider: string): boolean {
  return provider.trim().toLowerCase() === LOCAL_COSYVOICE3_PROVIDER
}

export function ttsProviderKind(option?: Pick<ComponentOption, 'tts_kind' | 'id'> | null): TTSProviderKind {
  if (option?.tts_kind?.trim()) return option.tts_kind.trim()
  if (option && isLocalCosyVoice3Provider(option.id)) return 'local_cosyvoice3'
  return 'cloud'
}

export function isLocalCosyVoice3TTS(
  provider: string,
  ttsComponent?: Pick<ComponentOption, 'tts_kind' | 'id'> | null,
): boolean {
  return ttsProviderKind(ttsComponent ?? { id: provider }) === 'local_cosyvoice3'
    || isLocalCosyVoice3Provider(provider)
}

/** Cloud DashScope CosyVoice models under the qwen TTS provider. */
export function isCloudQwenCosyVoiceTTS(provider: string, model: string): boolean {
  return provider === 'qwen' && isCosyVoiceTTSModel(model)
}

export interface TTSVoiceOption {
  label: string
  value: string
  labelEn?: string
}

export function configuredTTSVoices(option?: ComponentOption | null): TTSVoiceOption[] {
  if (!option) return []
  if (option.voices?.length) {
    return option.voices.map((voice) => ({
      value: voice.value,
      label: voice.label,
      labelEn: voice.label_en,
    }))
  }
  const value = (option.default_voice || '').trim()
  if (!value) return []
  return [{ value, label: localCosyVoice3VoiceLabel(value), labelEn: 'Local configured voice' }]
}

export function configuredTTSDefaultVoice(option?: ComponentOption | null): string {
  const voices = configuredTTSVoices(option)
  if (voices.length > 0) return voices[0].value
  return (option?.default_voice || '').trim()
}

export function isConfiguredLocalTTSVoice(option: ComponentOption | null | undefined, voice: string): boolean {
  const trimmed = voice.trim()
  if (!trimmed || !isLocalCosyVoice3TTS(option?.id || '', option)) return false
  return configuredTTSVoices(option).some((entry) => entry.value === trimmed)
}

export function findTTSComponent(
  catalog: ComponentsResponse | undefined,
  provider: string,
): ComponentOption | undefined {
  return catalog?.tts.find((item) => item.id === provider)
}