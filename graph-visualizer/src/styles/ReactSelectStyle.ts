/**
 * @file ReactSelectStyle.ts
 * @description Custom design tokens and styling generators for React-Select dropdown components.
 * Provides unified dark-theme styling, hover states, and dynamic red-highlighted error states.
 */

/**
 * Base dark-theme stylesheet overrides for React-Select components.
 */
export const reactSelectCustomStyles = {

    singleValue: (provided: any) => ({
        ...provided,
        color: 'white',
    }),
    color: 'white',
    menu: (provided: any) => ({
        ...provided,
        backgroundColor: '#2c2a30',
        border: '2px solid rgb(95, 92, 102)',
        '&:hover': {
            backgroundColor: '#3e3c46',
            border: '2px solid rgb(132, 124, 150)',
        },
    }),

    control: (provided: any) => ({
        ...provided,
        color: '#ffc400',
        backgroundColor: '#2c2a30',
        border: '2px solid rgb(95, 92, 102)',

        '&:hover': {
            backgroundColor: '#3e3c46',
            border: '2px solid rgb(132, 124, 150)',
        },
    }),
    
    option: (provided: any) => ({
        ...provided,
        color: 'white', 
        backgroundColor: '#2c2a30',
        '&:hover': {
            backgroundColor: '#3e3c46',
        },
        margin: '0px',
    }),
};

export const getReactSelectStyles = (hasError: boolean = false) => ({
    ...reactSelectCustomStyles,
    control: (provided: any) => ({
        ...provided,
        color: '#ffc400',
        backgroundColor: hasError ? '#321d23' : '#2c2a30',
        border: hasError ? '2px solid #ff4d4f' : '2px solid rgb(95, 92, 102)',
        boxShadow: hasError ? '0 0 4px #ff4d4f' : provided.boxShadow,
        '&:hover': {
            backgroundColor: hasError ? '#3a1f26' : '#3e3c46',
            border: hasError ? '2px solid #ff4d4f' : '2px solid rgb(132, 124, 150)',
        },
    }),
});