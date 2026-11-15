import { describe, expect, it } from 'vitest'
import { isDoubaoTTSVoiceType } from '../utils/voice'

describe('useDoubaoVoicePreview eligibility', () => {
  it('allows official Doubao 2.0 voice ids only', () => {
    expect(isDoubaoTTSVoiceType('zh_female_xiaohe_uranus_bigtts')).toBe(true)
    expect(isDoubaoTTSVoiceType('S_custom_clone_id')).toBe(false)
  })
})