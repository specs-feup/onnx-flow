/**
 * @file ThemeSelector.tsx
 * @description Dropdown selector component for switching Cytoscape visual themes
 * (default, copeland, nerv, mono, eyestrain, valentines, easteregg).
 */

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import ss1 from "@/styles/cytoscape/ss1.ts";
import defaultstylesheet from "@/styles/cytoscape/default.ts";
import copeland from "@/styles/cytoscape/copeland.ts";
import nge from "@/styles/cytoscape/nge.ts";
import valentines from "@/styles/cytoscape/valentines.ts";
import special from "@/styles/cytoscape/special.ts";
import mono from "@/styles/cytoscape/mono.ts";

/**
 * Properties for the ThemeSelector component.
 */
interface ThemeSelectorProps {
    /** Callback to set the active Cytoscape stylesheet */
    setStylesheet: (sheet: any) => void;
}

/**
 * Theme selector dropdown component.
 *
 * @param props - ThemeSelector properties
 * @returns JSX element containing the theme selector dropdown menu
 */
export default function ThemeSelector({
    setStylesheet,
}: ThemeSelectorProps) {

    return (
        <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
                <button>Themes</button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content className="DropdownMenuContent">
                <DropdownMenu.Item
                    className="DropdownMenuItem"
                    onSelect={() => setStylesheet(defaultstylesheet)}
                >
                    default
                </DropdownMenu.Item>
                <DropdownMenu.Item
                    className="DropdownMenuItem"
                    onSelect={() => setStylesheet(copeland)}
                >
                    copeland
                </DropdownMenu.Item>
                <DropdownMenu.Item className="DropdownMenuItem" onSelect={() => setStylesheet(nge)}>
                    nerv
                </DropdownMenu.Item>
                <DropdownMenu.Item
                    className="DropdownMenuItem"
                    onSelect={() => setStylesheet(mono)}
                >
                    mono
                </DropdownMenu.Item>
                <DropdownMenu.Item className="DropdownMenuItem" onSelect={() => setStylesheet(ss1)}>
                    eyestrain
                </DropdownMenu.Item>
                <DropdownMenu.Item
                    className="DropdownMenuItem"
                    onSelect={() => setStylesheet(valentines)}
                >
                    valentines
                </DropdownMenu.Item>
                <DropdownMenu.Item
                    className="DropdownMenuItem"
                    onSelect={() => setStylesheet(special)}
                >
                    easteregg
                </DropdownMenu.Item>
            </DropdownMenu.Content>
        </DropdownMenu.Root>
    );
}
