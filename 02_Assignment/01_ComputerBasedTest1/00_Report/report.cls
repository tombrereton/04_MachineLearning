\ProvidesClass{application}[2014/03/04 v1.0 Simple application class]
\LoadClass{article}

\RequirePackage[T1]{fontenc}
\RequirePackage{lmodern}
\renewcommand*\familydefault{\sfdefault}
\RequirePackage{sfmath}
\RequirePackage{microtype}
\RequirePackage[utf8]{inputenc}

\usepackage{cfr-lm}

\RequirePackage{listings}
\lstset{language=java}

%\RequirePackage[ngerman]{babel}
\RequirePackage[english]{babel}
%\selectlanguage{english}
%\selectlanguage{german}
\RequirePackage[margin=3.0cm,a4paper]{geometry}

\RequirePackage{graphicx}
\graphicspath{{./}{../}{images/}}

%\RequirePackage[pdftex,
	%unicode=true,
	%breaklinks=true,
	%colorlinks=true,
	%linkcolor=black,
	%citecolor=black,
	%urlcolor=black]{hyperref}

\renewcommand{\baselinestretch}{1.1}
\setlength{\parindent}{0pt}
%\setlength{\parskip}{5pt}

\newcommand{\titleabstand}{\hspace{3cm}}
\newcommand{\RNum}[1]{\uppercase\expandafter{\romannumeral #1\relax}}

\renewcommand{\maketitle}{
\begin{titlepage}
	\thispagestyle{empty}
%  \begin{minipage}[t]{1.0\textwidth}
%	\includegraphics[scale=0.17]{images/Birmingham_logo.png}
%  \end{minipage}
  \begin{minipage}[t]{1.0\textwidth}
  \vspace{0cm}
	\hfill
	\includegraphics[scale=0.18]{images/Birmingham_logo.png}
  \end{minipage}

 \begin{minipage}{\textwidth}
  \ \\
  \vspace{0.5cm}

  \begin{minipage}[t]{1.0\textwidth}
  	\begin{flushright}
  	\Large{Machine Learning (Extended)}\\
	\Huge{Computer Based Test 1 \\ Linear Regression by Least Squares}\\
	\Large \today
  	\end{flushright}
    \end{minipage}
  %\hspace{1.8cm}
\end{minipage}

\hspace{3cm}
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak
\linebreak



\Large
\begin{flushright}
Thomas Brereton 1708846\\
\end{flushright}

%\vspace{-3cm}
%\begin{flushleft}
%Number\\
%Number\\
%Number\\
%1708846\\
%\end{flushleft}

\end{titlepage}
}
