import React from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';


export default function CustomDropdown({ setLayout }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button>Graph Layout</button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content className="DropdownMenuContent">
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( {name:"fcose"})}>
          Default (fcose)
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({name: "breadthfirst"})}>
          BFS
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({name: "dagre", rankDir: "LR"})}>
          Left to Right
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( {name: "dagre", rankDir: "TB"})}>
          Top to Bottom
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({name: "grid"})}>
          Grid
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( {name:"concentric"})}>
          Circle
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}

