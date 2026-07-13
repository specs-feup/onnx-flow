import React from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';


export default function CustomDropdown({ setLayout }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button>Graph Layout</button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content className="DropdownMenuContent">
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( "fcose")}>
          Default
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout("breadthfirst")}>
          BFS
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( "")}>
          DFS (NOT DONE YET)
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( "concentric")}>
          Circle
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( "grid")}>
          Grid
        </DropdownMenu.Item>
      </DropdownMenu.Content>

    </DropdownMenu.Root>
  );
}

