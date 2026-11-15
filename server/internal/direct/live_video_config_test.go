package direct

import (
	"testing"

	"github.com/cyberverse/server/internal/config"
)

func TestConfigureLiveVideoAppliesCustomBitrateCaps(t *testing.T) {
	ConfigureLiveVideo(config.DirectVideoConfig{
		MaxBitrateKbps:   4000,
		MinBitrateKbps:   3000,
		GCCSafetyPercent: 95,
	})
	p := &DirectPeer{}
	p.targetBitrateBps.Store(10_000_000)
	if got := p.currentVideoBitrateKbps(); got != 4000 {
		t.Fatalf("expected max cap 4000, got %d", got)
	}
	ConfigureLiveVideo(config.OfficialDirectVideoDefaults())
}