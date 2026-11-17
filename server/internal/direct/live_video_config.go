package direct

import (
	"sync"

	"github.com/cyberverse/server/internal/config"
)

var (
	liveVideoMu  sync.RWMutex
	liveVideoCfg = config.OfficialDirectVideoDefaults()
)

// ConfigureLiveVideo applies pipeline.direct_video settings (call once at process start).
func ConfigureLiveVideo(cfg config.DirectVideoConfig) {
	liveVideoMu.Lock()
	defer liveVideoMu.Unlock()
	liveVideoCfg = cfg.WithDefaults()
}

func liveVideoConfig() config.DirectVideoConfig {
	liveVideoMu.RLock()
	defer liveVideoMu.RUnlock()
	return liveVideoCfg
}