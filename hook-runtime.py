"""
Runtime hook for Raptur macOS app.
Sets up environment variables so bundled ffmpeg/rubberband can find their dylibs.
Also patches PIL to avoid X11 library loading issues on macOS.
"""
import os
import sys

def _patch_pil_x11():
    """
    Patch PIL to prevent it from trying to load X11 libraries on macOS.
    This must run BEFORE PIL is imported.
    
    The issue: Pillow's _imaging.so is compiled with X11/libxcb support for
    screenshot functionality. On macOS, we exclude these libs from the bundle
    (they cause code signing issues), but PIL still tries to load them.
    
    Solution: Stub out the X11-dependent modules so PIL never attempts the load.
    """
    if sys.platform != 'darwin':
        return
    
    import importlib.abc
    import importlib.machinery
    
    class X11BlockerFinder(importlib.abc.MetaPathFinder):
        """Block imports that would trigger X11 library loading."""
        
        BLOCKED_MODULES = {
            'PIL.ImageGrab',
        }
        
        def find_spec(self, fullname, path, target=None):
            if fullname in self.BLOCKED_MODULES:
                return importlib.machinery.ModuleSpec(
                    fullname,
                    X11BlockerLoader(),
                    is_package=False
                )
            return None
    
    class X11BlockerLoader(importlib.abc.Loader):
        """Return a stub module that raises NotImplementedError."""
        
        def create_module(self, spec):
            import types
            module = types.ModuleType(spec.name)
            module.__doc__ = "Stub module - X11 not available on macOS bundle"
            
            def grab(*args, **kwargs):
                raise NotImplementedError(
                    "Screenshot functionality not available in bundled macOS app"
                )
            
            module.grab = grab
            module.grabclipboard = grab
            return module
        
        def exec_module(self, module):
            pass
    
    sys.meta_path.insert(0, X11BlockerFinder())


def _setup_dyld_paths():
    """Configure DYLD paths for bundled native libraries."""
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
        macos_dir = os.path.dirname(sys.executable)
        frameworks_dir = os.path.join(os.path.dirname(macos_dir), 'Frameworks')
        
        lib_paths = [
            bundle_dir,
            macos_dir,
            frameworks_dir,
            os.path.join(bundle_dir, 'lib'),
        ]
        
        existing_paths = [p for p in lib_paths if os.path.exists(p)]
        
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        new_dyld = ':'.join(existing_paths)
        if current_dyld:
            new_dyld = f"{new_dyld}:{current_dyld}"
        os.environ['DYLD_LIBRARY_PATH'] = new_dyld
        
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = new_dyld
        
        current_path = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{bundle_dir}:{macos_dir}:{current_path}"


_patch_pil_x11()
_setup_dyld_paths()
