import { describe, expect, it } from 'vitest'
import type { ComponentOption } from '../types'
import {
  LOCAL_COSYVOICE3_DEFAULT_VOICE,
  configuredTTSDefaultVoice,
  configuredTTSVoices,
  isCloudQwenCosyVoiceTTS,
  isConfiguredLocalTTSVoice,
  isLocalCosyVoice3TTS,
} from './ttsProviderCapabilities'

const localOption: ComponentOption = {
  id: 'cosyvoice3',
  name: 'CosyVoice3',
  model: 'Fun-CosyVoice3-0.5B-2512',
  default: true,
  available: true,
  default_voice: LOCAL_COSYVOICE3_DEFAULT_VOICE,
  tts_kind: 'local_cosyvoice3',
  voices: [{ value: LOCAL_COSYVOICE3_DEFAULT_VOICE, label: 'Local default', label_en: 'Local default' }],
}

describe('ttsProviderCapabilities', () => {
  it('distinguishes local cosyvoice3 from cloud qwen CosyVoice models', () => {
    expect(isLocalCosyVoice3TTS('cosyvoice3', localOption)).toBe(true)
    expect(isCloudQwenCosyVoiceTTS('qwen', 'cosyvoice-v3-flash')).toBe(true)
    expect(isCloudQwenCosyVoiceTTS('cosyvoice3', 'Fun-CosyVoice3-0.5B-2512')).toBe(false)
  })

  it('resolves configured default voice for local provider', () => {
    expect(configuredTTSDefaultVoice(localOption)).toBe(LOCAL_COSYVOICE3_DEFAULT_VOICE)
    expect(configuredTTSVoices(localOption)).toHaveLength(1)
  })

  it('rejects unknown local voice ids', () => {
    expect(isConfiguredLocalTTSVoice(localOption, LOCAL_COSYVOICE3_DEFAULT_VOICE)).toBe(true)
    expect(isConfiguredLocalTTSVoice(localOption, 'longanyang')).toBe(false)
    expect(isConfiguredLocalTTSVoice(localOption, '温柔文雅')).toBe(false)
  })
})