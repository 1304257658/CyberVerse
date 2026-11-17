import { onBeforeUnmount, ref } from 'vue'
import { getDoubaoVoicePreviewMeta } from '../services/api'
import { isDoubaoTTSVoiceType } from '../utils/voice'
import { shouldCancelDoubaoVoicePreview } from '../utils/doubaoVoicePreviewToggle'

let sharedAudio: HTMLAudioElement | null = null
let previewGeneration = 0

function stopPlayback() {
  if (!sharedAudio) return
  sharedAudio.onended = null
  sharedAudio.onerror = null
  sharedAudio.pause()
  sharedAudio.currentTime = 0
  sharedAudio.removeAttribute('src')
  sharedAudio.load()
}

export function useDoubaoVoicePreview() {
  const previewingVoice = ref<string | null>(null)
  const previewError = ref('')

  function stopPreview() {
    previewGeneration++
    stopPlayback()
    previewingVoice.value = null
  }

  async function previewVoice(voiceType: string) {
    const trimmed = voiceType.trim()
    previewError.value = ''
    if (!trimmed) return
    if (!isDoubaoTTSVoiceType(trimmed)) {
      previewError.value = 'preview_not_official'
      return
    }

    if (shouldCancelDoubaoVoicePreview(previewingVoice.value, trimmed)) {
      stopPreview()
      return
    }

    stopPlayback()
    previewingVoice.value = trimmed
    const generation = previewGeneration
    try {
      const meta = await getDoubaoVoicePreviewMeta(trimmed)
      if (generation !== previewGeneration || previewingVoice.value !== trimmed) {
        return
      }
      if (!sharedAudio) {
        sharedAudio = new Audio()
      }
      sharedAudio.onended = () => {
        if (previewingVoice.value === trimmed) {
          previewingVoice.value = null
        }
      }
      sharedAudio.onerror = () => {
        previewingVoice.value = null
        previewError.value = 'preview_playback_failed'
      }
      sharedAudio.src = meta.play_url
      await sharedAudio.play()
      if (generation !== previewGeneration || previewingVoice.value !== trimmed) {
        stopPlayback()
      }
    } catch (err) {
      if (generation === previewGeneration) {
        previewingVoice.value = null
        previewError.value = err instanceof Error ? err.message : 'preview_failed'
      }
    }
  }

  onBeforeUnmount(() => {
    stopPreview()
  })

  return {
    previewingVoice,
    previewError,
    previewVoice,
    stopPreview,
  }
}