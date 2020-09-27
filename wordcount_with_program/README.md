## CMD 화면에서 현재 폴더(wordcount_with_program)로 이동

  C> wordcount.exe text.txt

  wordlist.dat, wordlist.txt, out.txt 확인



## Example: 매우 큰 파일(위키 말뭉치 등 수천만 라인)에 대한 word count 방법

   C> split.exe -4m input.txt  --> xaa, xab, xac, ... 등으로 분할

        // input.txt를 4백만 라인씩 여러 개의 파일로 분할

   C> wordcount.exe -i xaa

        //첫번째 파일 xaa에 대한 wordcount : "-i" 옵션 사용

   C> wordcount.exe -i -add xab

   C> wordcount.exe -i -add xac
