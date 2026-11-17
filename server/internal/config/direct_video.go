package config

// DirectVideoConfig controls Direct WebRTC live VP8 encode caps and GCC bounds.
type DirectVideoConfig struct {
	TargetBitrateKbps int `yaml:"target_bitrate_kbps,omitempty"`
	MinBitrateKbps    int `yaml:"min_bitrate_kbps,omitempty"`
	MaxBitrateKbps    int `yaml:"max_bitrate_kbps,omitempty"`
	GCCSafetyPercent  int `yaml:"gcc_safety_percent,omitempty"`

	VP8Deadline            string `yaml:"vp8_deadline,omitempty"`
	VP8CPUUsed             int    `yaml:"vp8_cpu_used,omitempty"`
	VP8FallbackBitrateKbps int    `yaml:"vp8_fallback_bitrate_kbps,omitempty"`

	GCCInitialBitrateBps int `yaml:"gcc_initial_bitrate_bps,omitempty"`
	GCCMinBitrateBps     int `yaml:"gcc_min_bitrate_bps,omitempty"`
	GCCMaxBitrateBps     int `yaml:"gcc_max_bitrate_bps,omitempty"`
}

// OfficialDirectVideoDefaults match the legacy hardcoded Direct WebRTC behavior.
func OfficialDirectVideoDefaults() DirectVideoConfig {
	return DirectVideoConfig{
		TargetBitrateKbps:      1800,
		MinBitrateKbps:         500,
		MaxBitrateKbps:         1800,
		GCCSafetyPercent:       65,
		VP8Deadline:            "realtime",
		VP8CPUUsed:             8,
		VP8FallbackBitrateKbps: 2000,
		// Effective values used by cmd/cyberverse-server for direct streaming (not webrtcapi zero-fill).
		GCCInitialBitrateBps: 2_500_000,
		GCCMinBitrateBps:     800_000,
		GCCMaxBitrateBps:     4_000_000,
	}
}

func (c DirectVideoConfig) WithDefaults() DirectVideoConfig {
	d := OfficialDirectVideoDefaults()
	if c.TargetBitrateKbps > 0 {
		d.TargetBitrateKbps = c.TargetBitrateKbps
	}
	if c.MinBitrateKbps > 0 {
		d.MinBitrateKbps = c.MinBitrateKbps
	}
	if c.MaxBitrateKbps > 0 {
		d.MaxBitrateKbps = c.MaxBitrateKbps
	}
	if c.GCCSafetyPercent > 0 {
		d.GCCSafetyPercent = c.GCCSafetyPercent
	}
	if c.VP8Deadline != "" {
		d.VP8Deadline = c.VP8Deadline
	}
	if c.VP8CPUUsed > 0 {
		d.VP8CPUUsed = c.VP8CPUUsed
	}
	if c.VP8FallbackBitrateKbps > 0 {
		d.VP8FallbackBitrateKbps = c.VP8FallbackBitrateKbps
	}
	if c.GCCInitialBitrateBps > 0 {
		d.GCCInitialBitrateBps = c.GCCInitialBitrateBps
	}
	if c.GCCMinBitrateBps > 0 {
		d.GCCMinBitrateBps = c.GCCMinBitrateBps
	}
	if c.GCCMaxBitrateBps > 0 {
		d.GCCMaxBitrateBps = c.GCCMaxBitrateBps
	}
	return d
}