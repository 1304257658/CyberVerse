import { describe, expect, it } from 'vitest'
import type { Character } from '../types'
import { LOCAL_COSYVOICE3_DEFAULT_VOICE } from './voice'
import { characterCardOfflineTTS, hasExplicitOfflineVideoTTS } from './offlineVideoTtsDefaults'

const baseCharacter = (): Character => ({
  id: 'c1',
  name: 'Demo',
  description: '',
  avatar_image: '',
  avatar_backend: 'local_image',
  use_face_crop: false,
  mode: 'standard',
  voice_provider: 'doubao',
  voice_type: 'zh_female_shuangkuaisisi_moon_bigtts',
  components: { llm: 'qwen', asr: 'qwen', tts: 'doubao', tts_model: 'seed-tts-2.0' },
  speaking_style: '',
  personality: '',
  welcome_message: '',
  system_prompt: '',
  tags: [],
  images: [],
  active_image: '',
  image_mode: 'fixed',
  created_at: '',
  updated_at: '',
})

describe('offlineVideoTtsDefaults', () => {
  it('characterCardOfflineTTS follows voice_provider and voice_type', () => {
    const resolved = characterCardOfflineTTS(baseCharacter())
    expect(resolved).toEqual({
      provider: 'doubao',
      model: 'seed-tts-2.0',
      voice: 'zh_female_shuangkuaisisi_moon_bigtts',
    })
  })

  it('hasExplicitOfflineVideoTTS is false until user saves offline override', () => {
    expect(hasExplicitOfflineVideoTTS(null)).toBe(false)
    expect(hasExplicitOfflineVideoTTS(undefined)).toBe(false)
    expect(hasExplicitOfflineVideoTTS({ provider: 'qwen', model: '', voice: '' })).toBe(true)
  })
})