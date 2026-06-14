% !Mode:: "TeX:UTF-8"
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{模版}[2024/09/14 Chinese Custom Beamer Class]
\LoadClass[10pt,envcountsect,compress]{beamer}
\usetheme{Madrid}
\usefonttheme[onlymath]{serif}
\RequirePackage{xeCJK}  % 用于支持中文
\RequirePackage{listings, indentfirst}
\RequirePackage[backend=biber,style=apa,defernumbers=true,sorting=none,backref=true]{biblatex}
\RequirePackage[linesnumbered,ruled,vlined]{algorithm2e}
\addbibresource{文本分类.bib}
\RequirePackage{hyperref, fontspec, xltxtra, bbding, beamerthemesplit, tcolorbox, etoolbox}  % 一些常用宏包
\RequirePackage{amsmath, amssymb, amsfonts, txfonts, latexsym, unicode-math}  % 数学公式
\RequirePackage{xcolor, tikz, epsfig, graphicx, subfigure, overpic, media9, caption, subcaption, animate, booktabs}  % 插图
\RequirePackage{times, palatino, ragged2e, blkarray, svg}
% 字体设置
\setmainfont{Palatino} % Lato, Roboto, Candara, Helvetica, Palatino Linotype 衬线字体
\setsansfont{Tahoma}  % 无衬线字体
\setmonofont{Consolas} % Consolas, Courier New, Monaco 等宽字体
\setCJKmainfont{SimSun}  % 中文衬线字体
\setCJKsansfont{Hiragino Sans GB}  % 中文无衬线字体
\setCJKmonofont{FangSong}  % 中文等宽字体
\setbeamerfont{title}{series=\bfseries, size=\Large, family=\rmfamily}  % 标题:粗体,large,衬线字体
\setbeamerfont{subtitle}{size=\small, family=\rmfamily}  % 副标题:small,衬线字体
\setbeamerfont{author}{size=\large, family=\ttfamily}  % 作者:large,等宽字体
\setbeamerfont{institute}{size=\small, family=\sffamily}  % 机构:small,无衬线字体
\setbeamerfont{date}{size=\small, family=\rmfamily}  % 日期:small,衬线字体
\setbeamerfont{frametitle}{size=\large, family=\sffamily}  % 标题:large,无衬线字体
\setbeamerfont{section in head/foot}{size=\small, family=\sffamily}  % 标题:small,无衬线字体
\setbeamerfont{normal text}{family=\rmfamily, size=\normalsize}  % 正文字体:衬线字体,正常大小
\setbeamerfont{block title}{size=\large, family=\sffamily}  % 块标题:large,无衬线字体
\setbeamerfont{block body}{size=\normalsize, family=\rmfamily}  % 块正文字体:正常大小,衬线字体
\setbeamerfont{caption}{size=\small, family=\rmfamily}  % 图表标题:small,衬线字体
\setbeamerfont{caption name}{size=\small, family=\rmfamily}  % 图表标题:small,衬线字体
\setmathfont{Latin Modern Math}[
    Scale=MatchUppercase
]
% 重定义 bibfootnotetext 的样式，减小脚注字体
\providecommand*{\bibfootnotetext}[1]{\footnotetext{\tiny#1}}
\AtEveryCitekey{%
  \renewcommand*{\bibfootnotetext}[1]{\footnotetext{\tiny##1}}%
}
% 设置段落
\setlength{\parindent}{2em}  % 设置首行缩进为2em
%\renewcommand{\raggedright}{\leftskip=0pt \rightskip=0pt plus 0cm}
%\raggedright
% 设置参考文献
\setbeamertemplate{bibliography item}[text]
% 设置编号,页眉页脚
\setbeamertemplate{theorems}[numbered]
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{headline}{}
\setbeamercovered{invisible}  % 隐藏未显示的内容,也可设为transparent或dynamic
%------------------------------------------------------
% 自定义定理样式
\newtheoremstyle{chinese} % 新样式名为"chinese"
  {3pt}                    % 定理的上下间距
  {3pt}                    % 定理标题与内容的间距
  {\normalfont}            % 正文字体
  {}                       % 标题前的缩进
  {\bfseries}              % 定理标题加粗
  {.}                       % 定理标题后的标点符号
  {0.5em}                  % 标题后与正文的间距
  {#1~#2}                  % 定理的标题格式（#1为定理名称，#2为编号）
% 应用自定义样式
\theoremstyle{chinese}
% 定义中文环境
\newtheorem{mydefinition}{定义}[section]  % 修改为mydefinition，避免与LaTeX默认冲突
\newtheorem{mytheorem}{定理}[section]     % 修改为mytheorem，避免与LaTeX默认冲突
\newtheorem{mylemma}{引理}[section]       % 修改为mylemma，避免与LaTeX默认冲突
\newtheorem{myexample}{示例}[section]     % 修改为myexample，避免与LaTeX默认冲突
\newtheorem{myremark}{注释}[section]      % 修改为myremark，避免与LaTeX默认冲突
\newtheorem{mycorollary}{推论}[section]   % 修改为mycorollary，避免与LaTeX默认冲突
\newtheorem{myproperty}{性质}[section]   % 修改为myproperty，避免与LaTeX默认冲突
\newcommand{\handr}{\textcolor{magenta}{\HandRight}} % 自定义\handr
\newcommand{\jieda}{\noindent\textcolor{red}{\textbf {解答:}\quad}} % 自定义解答
\newcommand{\zheng}{\noindent\textcolor{red}{\textbf {证明:}\quad}}    % 自定义证明
\newcommand{\heiti}{\CJKfontspec{Heiti SC}}
\newcommand{\yunhan}{\,{\color{red!70}\Rightarrow}\,}   % 自定义推导
\renewcommand{\theequation}{\thesection.\arabic{equation}}% 方程的编号
% 定义日期
\renewcommand{\today}{\number\year-\number\month-\number\day}
% 设置标题页
\makeatletter
\let\@afterindentfalse\@afterindenttrue
\@afterindenttrue
\makeatother
% 在每个 frame 中默认使用 allowframebreaks 选项, 以允许帧内容自动分页(消除 Overfull \vbox 警告)
\let\oldframe\frame
\let\endoldframe\endframe
\renewenvironment{frame}[1][allowframebreaks]{\oldframe[#1]}{\endoldframe}
\lstset{
    language=Python,
    basicstyle=\fontspec{Noto Sans Mono}\scriptsize,  % 设置代码字体
    keywordstyle=\color{blue},              % 设置关键字颜色
    commentstyle=\color{purple},      % 设置注释颜色
    stringstyle=\color{black},     % 设置字符串颜色
    backgroundcolor=\color{white},          % 设置背景颜色为白色
    frame=single,                           % 绘制单线框架
    xleftmargin=0.5em,                      % 设置左边距
    xrightmargin=0em,                       % 设置右边距
    rulecolor=\color{darkgray},             % 设置框架线颜色为深灰
    stepnumber=1,                           % 每行都显示行号
    numbers=left,                           % 将行号显示在左侧
    numberstyle=\tiny\color{black},         % 设置行号的格式
    numbersep=5pt,                          % 设置行号和代码之间的间距
    breaklines=true,                        % 允许自动换行
    breakatwhitespace=true,                 % 仅在空格处换行
    columns=flexible,                       % 设置列宽度自动调整
    escapeinside={*@}{@*},                  % 允许使用特定符号插入LaTeX命令
    showstringspaces=false,                 % 不显示字符串中的空格
    showspaces=false,                       % 不显示空格
    showtabs=false,                         % 不显示制表符
    tabsize=2,                              % 设置制表符宽度为2个空格
    extendedchars=true,                     % 允许使用扩展字符
    alsoletter={.},                         % 将句点视为字母
    sensitive=true,
    keywords={
        True, False, None, and, as, assert, async, await, break, class, continue, def, del,
        elif, else, except, finally, for, from, global, if, import, in, is, lambda, nonlocal,
        not, or, pass, raise, return, try, while, with, yield, list, dict, set, tuple, range, zip
    },
    alsoother={=, +, -, *, /, //, **, ==, !=, <, <=, >, >=, +=, -=, *=, /=, **=, //=, \%},
    linewidth=\textwidth,                   % 设置行宽为文本宽度
    breakautoindent=true,                   % 保持缩进
    keepspaces=true,                        % 保持空格
    moredelim=[is][\bfseries\color{green}]{~}{~},  % 设置特殊字符的颜色
}
\setbeamercolor{author in head/foot}{fg=white,bg=blue!50}
\setbeamercolor{title in head/foot}{fg=black,bg=gray!30}
\setbeamercolor{date in head/foot}{fg=white,bg=blue!50}
\setbeamertemplate{footline}{
  \leavevmode%
  \hbox{%
    \begin{beamercolorbox}[wd=.1\paperwidth,ht=2.5ex,dp=1.2ex,center]{author in head/foot}%
      \usebeamerfont{author in head/foot}\insertshortauthor
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.8\paperwidth,ht=2.5ex,dp=1.2ex,center]{title in head/foot}%
      \usebeamerfont{title in head/foot}\insertshorttitle
    \end{beamercolorbox}%
    \begin{beamercolorbox}[wd=.1\paperwidth,ht=2.5ex,dp=1.2ex,center]{date in head/foot}%
      \usebeamerfont{date in head/foot}\insertframenumber{} / \inserttotalframenumber
    \end{beamercolorbox}%
  }%
}
\renewcommand{\lstlistingname}{Code}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\setbeamertemplate{caption}[numbered]  % 若你用 Beamer
\hypersetup{
  bookmarks=true,         % 开启 PDF 书签
  unicode=true,           % 中文书签需要
  bookmarksopen=true,     % 默认展开书签
  bookmarksdepth=3        % 控制书签深度，1=section，2=subsection...
}
