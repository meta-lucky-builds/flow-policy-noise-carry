#!/usr/bin/env bash
# Makes LIBERO rendering work on this machine, trying the least invasive fix first:
#   1. GPU rendering as the container provides it
#   2. register NVIDIA's EGL library if it exists but has no vendor file
#   3. install NVIDIA's graphics libraries from the driver package matching this machine's driver
#   4. CPU rendering through OSMesa
# Writes the working backend to .render_backend, which run.py reads and records in the results.
set -uo pipefail

try_backend() {
  local backend=$1 err
  err=$(mktemp)
  if MUJOCO_GL=$backend PYOPENGL_PLATFORM=$backend timeout 300 python render_test.py </dev/null 2>"$err"; then
    rm -f "$err"
    return 0
  fi
  echo "   $backend failed: $(grep -E 'Error|error|Cannot|failed' "$err" | head -3 | cut -c1-200 | tr '\n' ' ')"
  rm -f "$err"
  return 1
}

write_vendor_file() {
  mkdir -p /usr/share/glvnd/egl_vendor.d
  cat > /usr/share/glvnd/egl_vendor.d/10_nvidia.json <<'JSON'
{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}
JSON
}

install_driver_graphics_libs() {
  local ver dir libdir url lib base
  ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')
  [ -n "$ver" ] || return 1
  echo "   driver version $ver"
  dir=$(mktemp -d)
  libdir=/usr/lib/x86_64-linux-gnu
  for url in "https://us.download.nvidia.com/XFree86/Linux-x86_64/$ver/NVIDIA-Linux-x86_64-$ver.run" \
             "https://us.download.nvidia.com/tesla/$ver/NVIDIA-Linux-x86_64-$ver.run"; do
    echo "   downloading $url"
    if curl -fsSL -o "$dir/driver.run" "$url"; then
      break
    fi
    rm -f "$dir/driver.run"
  done
  [ -f "$dir/driver.run" ] || { echo "   no matching driver package found"; rm -rf "$dir"; return 1; }
  # Extraction only. This installs no kernel module and does not touch the host driver.
  sh "$dir/driver.run" --extract-only --target "$dir/x" >/dev/null 2>&1 || { echo "   could not extract driver package"; rm -rf "$dir"; return 1; }
  for base in libEGL_nvidia libGLESv2_nvidia libGLESv1_CM_nvidia libGLX_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-gpucomp libnvidia-tls libnvidia-glvkspirv; do
    lib="$dir/x/$base.so.$ver"
    if [ -f "$lib" ] && [ ! -e "$libdir/$base.so.$ver" ]; then
      cp "$lib" "$libdir/"
    fi
  done
  ln -sf "libEGL_nvidia.so.$ver" "$libdir/libEGL_nvidia.so.0"
  ln -sf "libGLX_nvidia.so.$ver" "$libdir/libGLX_nvidia.so.0"
  ln -sf "libGLESv2_nvidia.so.$ver" "$libdir/libGLESv2_nvidia.so.2"
  ldconfig
  write_vendor_file
  rm -rf "$dir"
}

echo "--- rendering step 1: GPU rendering as provided"
if try_backend egl; then echo egl > .render_backend; echo "RENDER BACKEND: egl"; exit 0; fi

echo "--- rendering step 2: register an existing NVIDIA EGL library"
if ldconfig -p | grep -q libEGL_nvidia || find / -name 'libEGL_nvidia.so*' -not -path '/proc/*' 2>/dev/null | grep -q .; then
  write_vendor_file
  ldconfig
  if try_backend egl; then echo egl > .render_backend; echo "RENDER BACKEND: egl (vendor file added)"; exit 0; fi
else
  echo "   no NVIDIA EGL library present"
fi

echo "--- rendering step 3: install graphics libraries matching this driver"
if install_driver_graphics_libs && try_backend egl; then
  echo egl > .render_backend
  echo "RENDER BACKEND: egl (driver graphics libraries installed)"
  exit 0
fi

echo "--- rendering step 4: CPU rendering through OSMesa"
if try_backend osmesa; then
  echo osmesa > .render_backend
  echo "RENDER BACKEND: osmesa. Images come from a software renderer, and every method is compared on the same one."
  exit 0
fi

echo "RENDER FAILED: neither GPU nor CPU rendering works on this machine."
exit 1
