import React from 'react';
import * as DropdownMenu from '@radix-ui/react-dropdown-menu';
import ss1 from './styleSheets/ss1.ts';
import defaultstylesheet from './styleSheets/default.ts';
import copeland from './styleSheets/copeland.ts';
import nge from './styleSheets/nge.ts';

export default function Themess({setStylesheet}: {setStylesheet: (sheet: string) => void }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button>Themes</button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content className="DropdownMenuContent">
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setStylesheet(defaultstylesheet)}>
          default
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setStylesheet(ss1)}>
          tester
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setStylesheet(copeland)}>
          copeland
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setStylesheet(nge)}>
          nge
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}