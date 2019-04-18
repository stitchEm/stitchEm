// Copyright (c) 2012-2017 VideoStitch SAS
// Copyright (c) 2018 stitchEm

#include "shortcutdialog.hpp"
#include "ui_shortcutdialog.h"

#include <QShortcut>
#include <QAction>

ShortcutDialog::ShortcutDialog(QWidget *parent) : QDialog(parent), ui(new Ui::ShortcutDialog) {
  ui->setupUi(this);
  QList<QShortcut *> shortcuts = parent->findChildren<QShortcut *>();
  foreach (QShortcut *shortcut, shortcuts) {
#ifdef Q_OS_OSX
    // TODO FIXME: VSA-5721
    if (shortcut->key().toString() == "Ctrl+1" || shortcut->key().toString() == "Ctrl+2" ||
        shortcut->key().toString() == "Ctrl+3" || shortcut->key().toString() == "Ctrl+4") {
      continue;
    }
#endif

    if (!shortcut->whatsThis().isEmpty()) {
      QTreeWidgetItem *itemShortcut = new QTreeWidgetItem;
      itemShortcut->setText(0, shortcut->key().toString(QKeySequence::SequenceFormat::NativeText));
      itemShortcut->setText(1, shortcut->whatsThis());
      ui->shortcutTreeWidget->addTopLevelItem(itemShortcut);
    }
  }

  QList<const QAction *> actions = parent->findChildren<const QAction *>();
  foreach (const QAction *action, actions) {
    if (!action->shortcut().toString().isEmpty() && !action->text().isEmpty()) {
#ifdef Q_OS_OSX
      // TODO FIXME: VSA-3088
      if (action->shortcut().toString() == "F") {
        continue;
      }
#endif

      QTreeWidgetItem *itemShortcut = new QTreeWidgetItem;
      itemShortcut->setText(0, action->shortcut().toString(QKeySequence::SequenceFormat::NativeText));
      itemShortcut->setText(1, action->text());
      if (action->shortcut().toString() == "F5") {
        itemShortcut->setText(1, tr("Last opened project"));
      }
      if (action->shortcut().toString() == "F4") {
        itemShortcut->setText(1, tr("Last applied template"));
      }
#ifdef Q_OS_OSX
      // Replace the OS X quit app command name
      if (action->shortcut().toString() == "Alt+F4") {
        itemShortcut->setText(0,
                              (QKeySequence(Qt::CTRL + Qt::Key_Q)).toString(QKeySequence::SequenceFormat::NativeText));
      }
#endif
      ui->shortcutTreeWidget->addTopLevelItem(itemShortcut);
    }
  }
  setWindowTitle(tr("Shortcuts"));
}

ShortcutDialog::~ShortcutDialog() { delete ui; }
