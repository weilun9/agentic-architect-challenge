# SkyVault Technical FAQ

## Files are not syncing
Check that the desktop app shows "Up to date" in the system tray. If it shows "Paused", click Resume sync. Files larger than 50 GB, and file names containing the characters < > : " | ? *, are skipped. Signing out of the desktop app and signing back in rebuilds the sync index.

## Recovering a deleted file
Deleted files stay in Trash for 30 days on Personal and Pro plans and for 180 days on Business plans. Open Trash in the web app, select the file and choose Restore. Earlier versions of a file are listed under right-click > Version history.

## The desktop app is slow
Large first-time uploads are throttled to avoid saturating your connection. You can change the upload limit under Preferences > Network. Excluding folders such as node_modules or build output from sync also helps.

## Sharing links
A sharing link can be set to View or Edit, protected with a password, and given an expiry date. Business administrators can turn off public links for the whole organisation.

## Checking service status
Live service status and incident history are published at status.skyvault.example. Subscribing on that page sends incident updates by email.

## Supported systems
The desktop app supports Windows 10 and later, macOS 13 and later, and Ubuntu 22.04 and later. Mobile apps are available for iOS 16 and Android 10 or later.
