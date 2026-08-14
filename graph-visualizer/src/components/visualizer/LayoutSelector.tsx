/**
 * @file LayoutSelector.tsx
 * @description Dropdown menu component powered by Radix UI for switching graph layout algorithms.
 * Supported layouts include fcose (compound spring embedder), BFS, Dagre (LR/TB hierarchical),
 * Grid, Concentric (Circle), and ELK layered (LR/DOWN).
 */

import * as DropdownMenu from '@radix-ui/react-dropdown-menu';

/**
 * Properties for the LayoutSelector component.
 */
interface LayoutSelectorProps {
    /** Callback invoked with selected Cytoscape layout configuration object */
    setLayout: (layout: any) => void;
}

/**
 * Graph layout algorithm selector dropdown menu.
 *
 * @param props - LayoutSelector properties
 * @returns JSX element containing the dropdown menu
 */
export default function LayoutSelector({ setLayout }: LayoutSelectorProps) {

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
          Dagre (Left to Right)
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( {name: "dagre", rankDir: "TB"})}>
          Dagre (Top to Bottom)
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({name: "grid"})}>
          Grid
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout( {name:"concentric"})}>
          Circle
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({
            name: "elk", 
            elk: { 
                algorithm: 'layered', 
                'elk.direction': 'RIGHT' 
            }
        })}>
            ELK (Left to Right)
        </DropdownMenu.Item>
        <DropdownMenu.Item className='DropdownMenuItem' onSelect={() => setLayout({
            name: "elk", 
            elk: { 
                algorithm: 'layered', 
                'elk.direction': 'DOWN' 
            }
        })}>
            ELK (Top to Bottom)
        </DropdownMenu.Item>
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
}

