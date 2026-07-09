import Dropdown from 'react-bootstrap/Dropdown';

function BasicExample() {
  return (
    <Dropdown>
      <Dropdown.Toggle id="dropdown-basic">Graph Layout </Dropdown.Toggle>
      <Dropdown.Menu>
        <Dropdown.Item href="">BFS</Dropdown.Item>
        <Dropdown.Item href="">DFS</Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
  );
}
export default BasicExample;