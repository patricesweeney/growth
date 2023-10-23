export PATH="$HOME/.jenv/bin:$PATH"
eval "$(jenv init -)"



jenv doctor

brew install AdoptOpenJDK/openjdk/adoptopenjdk11

/usr/libexec/java_home -V

jenv add /Library/Java/JavaVirtualMachines/adoptopenjdk-11.jdk/Contents/Home

jenv versions

 file $(brew --prefix openjdk)/bin/java

 jenv global openjdk64-11.0.11