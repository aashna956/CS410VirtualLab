# [MeTA](https://meta-toolkit.org) Search App

This project is a very simple search engine that shows how you can use MeTA in
the backend of a web-based search application.

## Initial setup

### Dependencies

- [libjson-cpp-rpc](https://github.com/cinemast/libjson-rpc-cpp) needs to be
  installed. Any other dependencies that MeTA requires are also required.
- [Coffeescript](http://coffeescript.org/) is nice to use, and the project uses
  this to generate the Javascript files. If you don't like Coffeescript, you can
  slightly tweak the project to not use it.
- [NodeJS](https://nodejs.org/) is used for the proxy server.

### MeTA API Server

```bash
cd cpp/
git submodule update --init --recursive
cd meta/
mkdir build/
cd build/
```

Then, compile MeTA by using `cmake` and `make`. Detailed instructions are on the
MeTA site. We need to compile MeTA in order to use the static libraries that are
generated.

```bash
cd cpp/
mkdir meta-libs/
cp meta/build/src/*/*.a meta-libs/
cp meta/build/src/*/*/*.a meta-libs/
```

Now we can use the static libraries in the search application.

Next, we can compile the API server.

```bash
cd cpp/
mkdir build/
cd build/
CXX=clang++ cmake ../ -DCMAKE_BUILD_TYPE=Release
make -j4
```

The API server should now be built. You can run it from inside the `build/`
directory with the command

```bash
./api-server config.toml
```

After you make modifications to the source files, you should only have to run
`make` and restart the server.

### Web Frontend

```bash
cd web/
make
```

The `make` command compiles the Coffeescript into Javascript and places it in
the `javascript/` directory. You're now ready to start the server:

```bash
python2 -m SimpleHTTPServer
```

Whenever you modify the Coffeescript files, you'll need to run `make` and reload
the page.

### Node Proxy

The node proxy is needed for local development. It directs traffic to either the
C++ server or the frontend site itself. The following command sets up the
necessary node modules.

```bash
cd node/
npm install
```

Now, you can start the proxy server:


```bash
node proxy.js
```

View the page at `http://localhost:9001`, and enter a query! The first time you
run this, the index needs to be created. Subsequent times, it will simply be
loaded from disk just like in MeTA.
