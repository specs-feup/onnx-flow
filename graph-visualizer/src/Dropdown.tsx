import React from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

export default function CustomDropdown() {
  return (
    <DropdownMenu.Root>

      <DropdownMenu.Trigger asChild>
        <button >
          Graph Layout
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content className="DropdownMenuContent" sideOffset={5}>
          <DropdownMenu.Item className="DropdownMenuItem">
            BFS
          </DropdownMenu.Item>

          <DropdownMenu.Separator className="DropdownMenuSeparator" />

          <DropdownMenu.Item className="DropdownMenuItem">
            DFS
          </DropdownMenu.Item>
          <DropdownMenu.Arrow className="DropdownMenuArrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
