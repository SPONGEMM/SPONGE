; SPONGE NSIS Installer Script
; Auto-detects system language (Chinese / English)
; Parameterized by VARIANT (CPU, CUDA12, CUDA13, etc.)

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "WinVer.nsh"
!include "x64.nsh"
!include "FileFunc.nsh"
!include "WordFunc.nsh"

; ---------- Build-time defines (set by installer.ps1) ----------
; !define PRODUCT_VERSION "2.0.0.0"
; !define VARIANT         "CPU"
; !define STAGE_DIR       "release-artifacts\nsis\stage"
; !define OUTPUT_PATH     "release-artifacts\nsis\SPONGE-CPU-v2.0.0-installer.exe"
; !define LICENSE_FILE    "packaging\windows\license.rtf"

; ---------- Derived names ----------
!define PRODUCT_NAME    "SPONGE ${VARIANT}"
!define INSTALL_DIR     "SPONGE-${VARIANT}"
!define REG_KEY         "Software\SPONGE-${VARIANT}"
!define UNINSTALL_KEY   "Software\Microsoft\Windows\CurrentVersion\Uninstall\SPONGE-${VARIANT}"

; ---------- Installer attributes ----------
Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "${OUTPUT_PATH}"
InstallDir "$PROGRAMFILES64\${INSTALL_DIR}"
InstallDirRegKey HKLM "${REG_KEY}" "InstallDir"
RequestExecutionLevel admin
Unicode true

; ---------- Version info ----------
VIProductVersion "${PRODUCT_VERSION}"
VIAddVersionKey "FileVersion"     "${PRODUCT_VERSION}"
VIAddVersionKey "ProductName"     "${PRODUCT_NAME}"
VIAddVersionKey "ProductVersion"  "${PRODUCT_VERSION}"
VIAddVersionKey "CompanyName"     "SPONGE Development Team"
VIAddVersionKey "FileDescription" "${PRODUCT_NAME} Installer"
VIAddVersionKey "LegalCopyright"  "Copyright (c) 2022-2026 SPONGE Development Team"

; ---------- Multi-language ----------
!define MUI_LANGDLL_ALLLANGUAGES

; ---------- MUI pages ----------
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${LICENSE_FILE}"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; ---------- Languages (order matters: first = default) ----------
!insertmacro MUI_LANGUAGE "SimpChinese"
!insertmacro MUI_LANGUAGE "English"

; ---------- Language strings ----------
LangString SEC_MAIN_NAME    ${LANG_SIMPCHINESE} "${PRODUCT_NAME} 核心程序"
LangString SEC_MAIN_NAME    ${LANG_ENGLISH}     "${PRODUCT_NAME} Core"
LangString SEC_MAIN_DESC    ${LANG_SIMPCHINESE} "${PRODUCT_NAME} 核心程序和运行时文件。"
LangString SEC_MAIN_DESC    ${LANG_ENGLISH}     "Core ${PRODUCT_NAME} binaries and runtime files."
LangString SEC_PATH_NAME    ${LANG_SIMPCHINESE} "将 ${PRODUCT_NAME} 加入 PATH"
LangString SEC_PATH_NAME    ${LANG_ENGLISH}     "Add ${PRODUCT_NAME} to PATH"
LangString SEC_PATH_DESC    ${LANG_SIMPCHINESE} "将安装目录加入系统 PATH 环境变量。"
LangString SEC_PATH_DESC    ${LANG_ENGLISH}     "Adds the installation directory to the system PATH."
LangString ARCH_ERROR       ${LANG_SIMPCHINESE} "${PRODUCT_NAME} 仅支持 64 位 Windows。"
LangString ARCH_ERROR       ${LANG_ENGLISH}     "${PRODUCT_NAME} requires 64-bit Windows."

; ---------- Installer init ----------
Function .onInit
  ${IfNot} ${RunningX64}
    MessageBox MB_OK|MB_ICONSTOP "$(ARCH_ERROR)"
    Abort
  ${EndIf}

  ; Auto-select language based on system locale
  !insertmacro MUI_LANGDLL_DISPLAY
FunctionEnd

; ---------- Sections ----------
Section "$(SEC_MAIN_NAME)" SecMain
  SectionIn RO ; required, cannot be unchecked

  SetOutPath "$INSTDIR"
  File /r "${STAGE_DIR}\*.*"

  ; Write uninstall info to registry
  WriteRegStr HKLM "${REG_KEY}" "InstallDir" "$INSTDIR"
  WriteRegStr HKLM "${UNINSTALL_KEY}" \
    "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "${UNINSTALL_KEY}" \
    "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegStr HKLM "${UNINSTALL_KEY}" \
    "Publisher" "SPONGE Development Team"
  WriteRegStr HKLM "${UNINSTALL_KEY}" \
    "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" \
    "NoModify" 1
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" \
    "NoRepair" 1

  ; Estimate installed size
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKLM "${UNINSTALL_KEY}" \
    "EstimatedSize" $0

  ; Create uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

Section "$(SEC_PATH_NAME)" SecPath
  ; Append INSTDIR to system PATH
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  ${If} $0 != ""
    StrCpy $0 "$0;$INSTDIR"
  ${Else}
    StrCpy $0 "$INSTDIR"
  ${EndIf}
  WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" "$0"

  ; Notify running processes of environment change
  SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
SectionEnd

; ---------- Section descriptions ----------
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} "$(SEC_MAIN_DESC)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SecPath} "$(SEC_PATH_DESC)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ---------- Uninstaller ----------
Section "Uninstall"
  ; Remove PATH entry
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path"
  ${If} $0 != ""
    ; Remove our directory from PATH (with and without trailing semicolon)
    ${WordReplace} $0 ";$INSTDIR" "" "+" $0
    ${WordReplace} $0 "$INSTDIR;" "" "+" $0
    ${WordReplace} $0 "$INSTDIR"  "" "+" $0
    WriteRegExpandStr HKLM "SYSTEM\CurrentControlSet\Control\Session Manager\Environment" "Path" "$0"
    SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}

  ; Remove files and directory
  RMDir /r "$INSTDIR"

  ; Remove registry keys
  DeleteRegKey HKLM "${UNINSTALL_KEY}"
  DeleteRegKey HKLM "${REG_KEY}"
SectionEnd
