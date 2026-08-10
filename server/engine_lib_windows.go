//go:build windows

package main

import (
	"fmt"
	"syscall"
)

// openLib loads a DLL via LoadLibrary on Windows. The returned uintptr is
// the HMODULE, suitable for purego.RegisterLibFunc — purego's loadSymbol on
// Windows uses syscall.GetProcAddress(syscall.Handle(handle), name).
func openLib(path string) (uintptr, error) {
	h, err := syscall.LoadLibrary(path)
	if err != nil {
		return 0, fmt.Errorf("LoadLibrary %s: %w", path, err)
	}
	return uintptr(h), nil
}
