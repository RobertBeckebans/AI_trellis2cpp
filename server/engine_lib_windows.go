//go:build windows

package main

import (
	"fmt"
	"syscall"
	"unsafe"
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

// setNativeEnv makes a variable visible to getenv() inside the loaded library.
//
// os.Setenv is not enough on Windows: it calls SetEnvironmentVariableW, which
// updates the process environment block, while the UCRT that the library's
// getenv() reads keeps its own copy taken when the CRT started. Anything set
// after process start is therefore invisible to the library — the reason the
// first attempt at disabling CUDA graphs from main() changed nothing and the
// server kept dying in graph capture. Poking ucrtbase's own _putenv reaches the
// copy that actually gets read.
//
// Safe to call before or after the library is loaded, as long as it happens
// before the variable is first read; ggml reads this one lazily on the first
// graph compute.
func setNativeEnv(name, value string) error {
	crt, err := syscall.LoadLibrary("ucrtbase.dll")
	if err != nil {
		return fmt.Errorf("LoadLibrary ucrtbase.dll: %w", err)
	}
	proc, err := syscall.GetProcAddress(crt, "_putenv")
	if err != nil {
		return fmt.Errorf("GetProcAddress _putenv: %w", err)
	}
	s, err := syscall.BytePtrFromString(name + "=" + value)
	if err != nil {
		return err
	}
	r, _, _ := syscall.SyscallN(uintptr(proc), uintptr(unsafe.Pointer(s)))
	if int32(r) != 0 {
		return fmt.Errorf("_putenv %s failed", name)
	}
	return nil
}
