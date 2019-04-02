import lyricsgenius
genius = lyricsgenius.Genius("Fm04q4dMHt4JIsiAgMYfxsvLfK5vhuP3FOoh8Ts_gOpPahaVvdAUEyOSkEldxZHs")
artist = genius.search_artist("Kanye West")
artist.save_lyrics()
