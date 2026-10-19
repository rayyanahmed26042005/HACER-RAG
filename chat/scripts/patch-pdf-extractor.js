const fs = require('fs');
const path = require('path');

const targetFile = path.join(
  __dirname,
  '..',
  'node_modules',
  'react-native-pdf-extractor',
  'android',
  'src',
  'main',
  'kotlin',
  'com',
  'reactnative',
  'pdf',
  'extractor',
  'PdfExtractorModule.kt'
);

if (fs.existsSync(targetFile)) {
  let content = fs.readFileSync(targetFile, 'utf8');
  if (content.includes('fun canIExtract(promise: Promise): Any {')) {
    content = content.replace(
      'fun canIExtract(promise: Promise): Any {',
      'fun canIExtract(promise: Promise) {'
    );
    content = content.replace(
      'return promise.resolve(resolver.getType(uri).equals("application/pdf"))',
      'promise.resolve(resolver.getType(uri).equals("application/pdf")); return'
    );
    content = content.replace(
      'return promise.resolve(false)',
      'promise.resolve(false)'
    );
    content = content.replace(
      'return promise.reject(e)',
      'promise.reject(e)'
    );
    fs.writeFileSync(targetFile, content, 'utf8');
    console.log('[patch-pdf-extractor] Successfully patched PdfExtractorModule.kt for TurboModules/JNI.');
  }
}

const targetGradle = path.join(
  __dirname,
  '..',
  'node_modules',
  'react-native-pdf-extractor',
  'android',
  'build.gradle'
);

if (fs.existsSync(targetGradle)) {
  let gradleContent = fs.readFileSync(targetGradle, 'utf8');
  if (gradleContent.includes('minifyEnabled true')) {
    gradleContent = gradleContent.replace('minifyEnabled true', 'minifyEnabled false');
    fs.writeFileSync(targetGradle, gradleContent, 'utf8');
    console.log('[patch-pdf-extractor] Successfully disabled minifyEnabled in react-native-pdf-extractor build.gradle.');
  }
}

