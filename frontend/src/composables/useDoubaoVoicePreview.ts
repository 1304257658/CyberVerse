import { onBeforeUnmount, ref } from 'vue'
import { getDoubaoVoicePreviewMeta } from '../services/api'
import { isDoubaoTTSVoiceType } from '../utils/voice'

let sharedAudio: HTMLAudioElement | null = null

function stopPlayback() {
  if (!sharedAudio) return
  sharedAudio.pause()
  sharedAudio.currentTime = 0
  sharedAudio.removeAttribute('src')
  sharedAudio.load()
}

export function useDoubaoVoicePreview() {
  const previewingVoice = ref<string | null>(null)
  const previewError = ref('')

  async function previewVoice(voiceType: string) {
    const trimmed = voiceType.trim()
    previewError.value = ''
    if (!trimmed) return
    if (!isDoubaoTTSVoiceType(trimmed)) {
      previewError.value = 'preview_not_official'
      return
    }
    stopPlayback()
    if (previewingVoice.value === trimmed && sharedAudio && !sharedAudio.paused) {
      previewingVoice.value = null
      return
    }
    previewingVoice.value = trimmed
    try {
      const meta = await getDoubaoVoicePreviewMeta(trimmed)
      if (!sharedAudio) {
        sharedAudio = new Audio()
      }
      sharedAudio.onended = () => {
        previewingVoice.value = null
      }
      sharedAudio.onerror = () => {
        previewingVoice.value = null
        previewError.value = 'preview_playback_failed'
      }
      sharedAudio.src = meta.play_url
      await sharedAudio.play()
    } catch (err) {
      previewingVoice.value = null
      previewError.value = err instanceof Error ? err.message : 'preview_failed'
    }
  }

  function stopPreview() {
    stopPlayback()
    previewingVoice.value = null
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