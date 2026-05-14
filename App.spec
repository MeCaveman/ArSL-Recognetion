# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['App.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('hand_signs', 'hand_signs'),
        ('runs', 'runs'),
        ('config', 'config'),
        ('src', 'src')
    ],
    # Ultralytics and PySide6 often need to be explicitly declared
    hiddenimports=['PySide6', 'ultralytics', 'cv2'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ArSL_Recognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ArSL_Recognition',
)
