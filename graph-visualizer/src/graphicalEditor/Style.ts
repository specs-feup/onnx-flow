export const reactSelectCustomStyles = {
    singleValue: (provided: any, state: any) => ({
        ...provided,
        color: 'white',
    }),
    color: 'white',
    menu: (provided: any, state: any) => ({
        ...provided,
        backgroundColor: '#2c2a30',
        border: '2px solid rgb(95, 92, 102)',
        '&:hover': {
            backgroundColor: '#3e3c46',
            border: '2px solid rgb(132, 124, 150)',
            },
        }),

    control: (provided: any, state: any) => ({
      ...provided,
      color: '#ffc400',
      backgroundColor: '#2c2a30',
      border: '2px solid rgb(95, 92, 102)',

      '&:hover': {
        backgroundColor: '#3e3c46',
        border: '2px solid rgb(132, 124, 150)',
      },
    }),
    
    option: (provided: any, state: any) => ({
        ...provided,
        color: 'white', 
        backgroundColor: '#2c2a30',
        '&:hover': {
        backgroundColor: '#3e3c46',

        },
    margin: '0px',
  }),
};