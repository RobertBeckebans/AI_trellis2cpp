//go:build !windows

package main

import "github.com/ebitengine/purego"

// openLib loads a shared library handle via dlopen on Unix-like platforms.
// The returned uintptr is suitable for purego.RegisterLibFunc.
func openLib(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

// setNativeEnv is only needed on Windows, where the CRT keeps its own copy of
// the environment; elsewhere os.Setenv already reaches the library's getenv().
func setNativeEnv(name, value string) error { return nil }
