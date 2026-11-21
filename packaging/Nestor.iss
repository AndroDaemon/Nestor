; Inno Setup script for Nestor
; Compiler: ISCC (Inno Setup Compiler)
;
; The version is read from pyproject.toml — the single source of truth. Do not
; duplicate it here. Everything else the wizard shows (icon, license, release
; notes, help pages) sits next to this file in packaging/.

#define MyAppName "Nestor"
#define MyAppVersion StringChange(StringChange(ReadIni(SourcePath + "..\pyproject.toml", "project", "version", "0.0.0"), """", ""), " ", "")
#define MyAppPublisher "Nestor"
#define MyAppExeName "Nestor.exe"
#define MyAppURL "https://github.com/AndroDaemon/Nestor"
#define MyAppMutex "Nestor.Desktop.SingleInstance"
#define InstallDir "{localappdata}\Nestor"

[Setup]
AppId={{8A1B2C3D-4E5F-6A7B-8C9D-0E1F2A3B4C5D}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={#InstallDir}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
DisableDirPage=auto
DisableProgramGroupPage=auto
OutputDir=output
OutputBaseFilename=Nestor-Setup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=nestor.ico
WizardStyle=modern dark
WizardImageFile=wizard-large.png, wizard-large@2x.png
WizardImageBackColor=#000000
WizardSmallImageFile=wizard-small.png
LicenseFile=license.txt
InfoBeforeFile=whatsnew.txt
CloseApplications=yes
AppMutex={#MyAppMutex}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
VersionInfoCopyright=Copyright (C) 2026 Nestor contributors

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; The application itself (PyInstaller onedir build).
Source: "..\dist\Nestor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Offline user guide, rendered from docs/rukovodstvo.md by build/_make_help.py.
Source: "help\*"; DestDir: "{app}\docs"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Руководство пользователя"; Filename: "{app}\docs\index.html"
Name: "{group}\Открыть папку с данными"; Filename: "{sys}\explorer.exe"; Parameters: """{app}"""
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, "&", "&&")}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"

[Code]
const
  DataPrompt =
    'Удалить расшифровки, голосовую библиотеку и настройки?' + #13#10 + #13#10 +
    'Да — стереть записи, расшифровки, голоса и настройки.' + #13#10 +
    'Нет — сохранить данные: они пригодятся при следующей установке.';

{ Nestor keeps its data next to the program (%LOCALAPPDATA%\Nestor): output/ with
  transcripts, voiceprints/ with the voice library, plus caches. The uninstaller
  only removes files it installed, so a silent upgrade never loses data. Ask
  explicitly before wiping the user's own material. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usUninstall then
  begin
    DataDir := ExpandConstant('{app}');
    if DirExists(DataDir) then
    begin
      if MsgBox(DataPrompt, mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
      begin
        DelTree(DataDir + '\output', True, True, True);
        DelTree(DataDir + '\voiceprints', True, True, True);
        DelTree(DataDir + '\peaks', True, True, True);
        DelTree(DataDir + '\uploads', True, True, True);
        DelTree(DataDir + '\embeddings', True, True, True);
        DelTree(DataDir + '\models', True, True, True);
        DelTree(DataDir + '\docs', True, True, True);
        DeleteFile(DataDir + '\settings.json');
        DeleteFile(DataDir + '\hotwords.yaml');
        DeleteFile(DataDir + '\log.txt');
        DeleteFile(DataDir + '\ui.log');
        DeleteFile(DataDir + '\instance.json');
      end;
    end;
  end;
end;
