//go:build !windows

package main

import "github.com/ebitengine/purego"

// openLib loads a shared library handle via dlopen on Unix-like platforms.
// The returned uintptr is suitable for purego.RegisterLibFunc.
func openLib(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}
