vendor フォルダ（歌詞の読みを調べる部品）

  Janome-0.5.0-py2.py3-none-any.whl
    日本語の文章を単語に分けて読みを調べる部品です。
    歌詞のふりがなと、1音ずつの区切りに使います。
    初めて歌詞を入れたときに、ソフトのフォルダの _lib に自動で展開します。

    Janome   https://github.com/mocobeta/janome
      Copyright 2015 moco_beta
      Licensed under the Apache License, Version 2.0
      http://www.apache.org/licenses/LICENSE-2.0

    辞書 mecab-ipadic (IPA 辞書)
      Copyright 2000, 2001, 2002, 2003 Nara Institute of Science and Technology.
      くわしい条件は whl の中の NOTICE.txt / LICENSE.txt をご覧ください。

  openai_whisper-20250625-py3-none-any.zip
    歌声から歌詞と言葉ごとの時刻を聞き取る AI (OpenAI Whisper) の部品です。
    「歌詞 → AI で歌詞を聞き取る」を使ったときだけ、_lib に展開して使います。
    AI のモデル (large-v3-turbo / small) は、使うときに models\whisper へダウンロードします。
      OpenAI Whisper   https://github.com/openai/whisper
      Copyright (c) 2022 OpenAI   MIT License (zip の中の whisper-LICENSE.txt)

  tiktoken-0.14.0-cp312-cp312-win_amd64.whl   (Whisper が使う部品)
      https://github.com/openai/tiktoken   MIT License
  regex-2026.9.29-cp312-cp312-win_amd64.whl   (tiktoken が使う部品)
      https://github.com/mrabarnett/mrab-regex   Apache License 2.0 / CNRI-Python
