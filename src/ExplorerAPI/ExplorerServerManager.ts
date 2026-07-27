import * as fs from 'fs';
import * as path from 'path';

const folderPath: string = './examples/onnx';
const files: string[] = fs.readdirSync(folderPath);
const newfiles: string[] = files.filter((file: string) => path.extname(file) === '.onnx')

console.log('Directory contents:', newfiles);