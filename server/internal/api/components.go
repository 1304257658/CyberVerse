package api

import (
	"net/http"
	"sort"
	"strings"

	"github.com/cyberverse/server/internal/config"
	"gopkg.in/yaml.v3"
)

type componentVoiceOption struct {
	Value   string `json:"value"`
	Label   string `json:"label"`
	LabelEn string `json:"label_en,omitempty"`
}

type componentOption struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Model        string                 `json:"model"`
	Default      bool                   `json:"default"`
	Available    bool                   `json:"available"`
	DefaultVoice string                 `json:"default_voice,omitempty"`
	Voices       []componentVoiceOption `json:"voices,omitempty"`
	TTSKind      string                 `json:"tts_kind,omitempty"`
}

type componentsResponse struct {
	LLM []componentOption `json:"llm"`
	ASR []componentOption `json:"asr"`
	TTS []componentOption `json:"tts"`
}

func (r *Router) handleListComponents(w http.ResponseWriter, req *http.Request) {
	llm, err := r.configuredComponentOptions("llm")
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	asr, err := r.configuredComponentOptions("asr")
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	tts, err := r.configuredComponentOptions("tts")
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, componentsResponse{
		LLM: llm,
		ASR: asr,
		TTS: tts,
	})
}

func (r *Router) defaultComponentsResponse() componentsResponse {
	return componentsResponse{
		LLM: []componentOption{{ID: r.pipelineDefault("llm"), Name: displayComponentName(r.pipelineDefault("llm")), Default: true, Available: true}},
		ASR: []componentOption{{ID: r.pipelineDefault("asr"), Name: displayComponentName(r.pipelineDefault("asr")), Default: true, Available: true}},
		TTS: []componentOption{{ID: r.pipelineDefault("tts"), Name: displayComponentName(r.pipelineDefault("tts")), Default: true, Available: true}},
	}
}

func (r *Router) pipelineDefault(category string) string {
	if r != nil && r.cfg != nil {
		switch category {
		case "llm":
			if r.cfg.Pipeline.DefaultLLM != "" {
				return r.cfg.Pipeline.DefaultLLM
			}
		case "asr":
			if r.cfg.Pipeline.DefaultASR != "" {
				return r.cfg.Pipeline.DefaultASR
			}
		case "tts":
			if r.cfg.Pipeline.DefaultTTS != "" {
				return r.cfg.Pipeline.DefaultTTS
			}
		}
	}
	return "qwen"
}

func (r *Router) configuredComponentOptions(category string) ([]componentOption, error) {
	if r.configPath == "" {
		switch category {
		case "llm":
			return r.defaultComponentsResponse().LLM, nil
		case "asr":
			return r.defaultComponentsResponse().ASR, nil
		case "tts":
			return r.defaultComponentsResponse().TTS, nil
		default:
			return nil, nil
		}
	}
	doc, err := config.ReadResolvedYAMLNode(r.configPath)
	if err != nil {
		return nil, err
	}
	return r.componentOptions(doc, category, r.pipelineDefault(category)), nil
}

func (r *Router) configuredTTSProvider(provider string) bool {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return false
	}
	options, err := r.configuredComponentOptions("tts")
	if err != nil {
		return false
	}
	for _, opt := range options {
		if opt.ID == provider && opt.Available {
			return true
		}
	}
	return false
}

func (r *Router) configuredTTSVoice(provider string) string {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		provider = r.pipelineDefault("tts")
	}
	if r.configPath != "" {
		if doc, err := config.ReadResolvedYAMLNode(r.configPath); err == nil {
			if section, err := config.GetNodeAtPath(doc, "inference.tts."+provider); err == nil {
				if voice := scalarAt(section, "voice"); voice != "" {
					return voice
				}
			}
		}
	}
	return defaultTTSVoice(provider)
}

func defaultTTSVoice(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "qwen":
		return "Momo"
	case "doubao":
		return "zh_female_xiaohe_uranus_bigtts"
	case "openai":
		return "nova"
	case "cosyvoice3":
		return "cyberverse_default"
	default:
		return ""
	}
}

func (r *Router) componentOptions(doc *yaml.Node, category string, fallbackDefault string) []componentOption {
	section, err := config.GetNodeAtPath(doc, "inference."+category)
	if err != nil || section.Kind != yaml.MappingNode {
		return []componentOption{{
			ID:        fallbackDefault,
			Name:      displayComponentName(fallbackDefault),
			Default:   true,
			Available: true,
		}}
	}

	defaultID := fallbackDefault
	if n := mappingValue(section, "default"); n != nil {
		if v := strings.TrimSpace(config.NodeScalarValue(n, true)); v != "" {
			defaultID = v
		}
	}

	options := make([]componentOption, 0)
	for i := 0; i < len(section.Content)-1; i += 2 {
		id := section.Content[i].Value
		if id == "default" {
			continue
		}
		node := section.Content[i+1]
		if node.Kind != yaml.MappingNode {
			continue
		}
		pluginClass := scalarAt(node, "plugin_class")
		if pluginClass == "" {
			continue
		}
		opt := componentOption{
			ID:        id,
			Name:      displayComponentName(id),
			Model:     componentModel(node, category),
			Default:   id == defaultID,
			Available: pluginClass != "",
		}
		if category == "tts" {
			enrichTTSComponentOption(&opt, id, pluginClass, node)
		}
		options = append(options, opt)
	}

	sort.SliceStable(options, func(i, j int) bool {
		if options[i].Default != options[j].Default {
			return options[i].Default
		}
		return options[i].ID < options[j].ID
	})
	return options
}

func mappingValue(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i < len(node.Content)-1; i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

func scalarAt(node *yaml.Node, key string) string {
	child := mappingValue(node, key)
	if child == nil {
		return ""
	}
	return strings.TrimSpace(config.NodeScalarValue(child, true))
}

func componentModel(node *yaml.Node, category string) string {
	for _, key := range []string{"model", "model_size"} {
		if value := scalarAt(node, key); value != "" {
			return value
		}
	}
	return category
}

func isLocalCosyVoice3TTSPlugin(providerID, pluginClass string) bool {
	providerID = strings.ToLower(strings.TrimSpace(providerID))
	pluginClass = strings.ToLower(strings.TrimSpace(pluginClass))
	return providerID == "cosyvoice3" || strings.Contains(pluginClass, "cosyvoice3")
}

func enrichTTSComponentOption(opt *componentOption, providerID, pluginClass string, node *yaml.Node) {
	if opt == nil || node == nil {
		return
	}
	voice := scalarAt(node, "voice")
	if voice != "" {
		opt.DefaultVoice = voice
	}
	if isLocalCosyVoice3TTSPlugin(providerID, pluginClass) {
		opt.TTSKind = "local_cosyvoice3"
		if voice == "" {
			voice = "cyberverse_default"
			opt.DefaultVoice = voice
		}
		opt.Voices = []componentVoiceOption{{
			Value:   voice,
			Label:   localCosyVoice3VoiceLabel(voice),
			LabelEn: "Local configured voice",
		}}
	}
}

func localCosyVoice3VoiceLabel(voiceID string) string {
	switch strings.TrimSpace(voiceID) {
	case "cyberverse_default":
		return "本地默认音色（zero-shot 参考）"
	default:
		return voiceID
	}
}

func (r *Router) configuredTTSModel(provider string) string {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return ""
	}
	options, err := r.configuredComponentOptions("tts")
	if err != nil {
		return ""
	}
	for _, opt := range options {
		if opt.ID == provider {
			return strings.TrimSpace(opt.Model)
		}
	}
	return ""
}

func (r *Router) configuredTTSVoiceAllowList(provider string) []string {
	provider = strings.TrimSpace(provider)
	if provider == "" {
		return nil
	}
	options, err := r.configuredComponentOptions("tts")
	if err != nil {
		return nil
	}
	for _, opt := range options {
		if opt.ID != provider {
			continue
		}
		if len(opt.Voices) > 0 {
			out := make([]string, 0, len(opt.Voices))
			for _, voice := range opt.Voices {
				if v := strings.TrimSpace(voice.Value); v != "" {
					out = append(out, v)
				}
			}
			return out
		}
		if v := strings.TrimSpace(opt.DefaultVoice); v != "" {
			return []string{v}
		}
		if v := defaultTTSVoice(provider); v != "" {
			return []string{v}
		}
	}
	return nil
}

func (r *Router) isAllowedTTSVoice(provider, voice string) bool {
	voice = strings.TrimSpace(voice)
	if voice == "" {
		return false
	}
	allow := r.configuredTTSVoiceAllowList(provider)
	if len(allow) == 0 {
		return true
	}
	for _, allowed := range allow {
		if allowed == voice {
			return true
		}
	}
	return false
}

func displayComponentName(id string) string {
	switch strings.ToLower(strings.TrimSpace(id)) {
	case "qwen":
		return "Qwen"
	case "cosyvoice3":
		return "CosyVoice3 (local)"
	case "openai":
		return "OpenAI"
	case "ollama":
		return "Ollama"
	case "atlas":
		return "Atlas Cloud"
	case "cheaperinference":
		return "Cheaper Inference"
	case "uberllm":
		return "UberLLM"
	case "runcrate":
		return "Runcrate"
	case "whisper":
		return "Whisper"
	case "windows_sapi":
		return "Windows Speech (local)"
	default:
		if id == "" {
			return "Qwen"
		}
		return strings.ToUpper(id[:1]) + id[1:]
	}
}
