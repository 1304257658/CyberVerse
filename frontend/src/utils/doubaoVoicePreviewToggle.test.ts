import { describe, expect, it } from 'vitest'
import { shouldCancelDoubaoVoicePreview } from './doubaoVoicePreviewToggle'

describe('shouldCancelDoubaoVoicePreview', () => {
  it('cancels when the same voice is already in preview mode', () => {
    expect(shouldCancelDoubaoVoicePreview('zh_female_xiaohe_uranus_bigtts', 'zh_female_xiaohe_uranus_bigtts')).toBe(true)
    expect(shouldCancelDoubaoVoicePreview('zh_female_xiaohe_uranus_bigtts', '  zh_female_xiaohe_uranus_bigtts  ')).toBe(true)
  })

  it('does not cancel when previewing a different voice', () => {
    expect(shouldCancelDoubaoVoicePreview('zh_female_vv_uranus_bigtts', 'zh_female_xiaohe_uranus_bigtts')).toBe(false)
  })

  it('does not cancel when idle', () => {
    expect(shouldCancelDoubaoVoicePreview(null, 'zh_female_xiaohe_uranus_bigtts')).toBe(false)
    expect(shouldCancelDoubaoVoicePreview(null, '')).toBe(false)
  })
})