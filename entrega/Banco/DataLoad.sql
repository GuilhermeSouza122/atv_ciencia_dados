INSERT INTO "User" (user_id, username) VALUES
(1, 'user01'),
(2, 'user02'),
(3, 'user03'),
(4, 'user04'),
(5, 'user05');

INSERT INTO "StreamingHistory" (
    user_id,
    played_at,
    platform_used,
    connection_country,
    track_id,
    artist_id,
    album_id,
    is_podcast,
    play_start_reason,
    play_end_reason,
    shuffle_enabled,
    skipped_track,
    milliseconds_played,
    offline_playback,
    offline_timestamp,
    incognito_mode
) VALUES
(1, '2026-02-24 10:12:35', 'mobile', 'BR', 101, 11, 201, FALSE, 'click', 'endplay', FALSE, FALSE, 185000, FALSE, NULL, FALSE),
(1, '2026-02-24 11:45:10', 'desktop', 'BR', 102, 12, 202, FALSE, 'autoplay', 'click', TRUE, FALSE, 223000, FALSE, NULL, FALSE),
(2, '2026-02-23 20:14:55', 'tablet', 'US', 103, 13, 203, FALSE, 'click', 'autoplay', FALSE, TRUE, 52000, FALSE, NULL, FALSE),
(2, '2026-02-23 21:03:12', 'mobile', 'US', 104, 13, 203, TRUE, 'click', 'endplay', FALSE, FALSE, 900000, TRUE, 1708740000, FALSE),
(3, '2026-02-22 08:55:43', 'desktop', 'BR', 105, 14, 204, FALSE, 'autoplay', 'endplay', TRUE, FALSE, 150000, FALSE, NULL, TRUE),
(3, '2026-02-22 09:20:10', 'mobile', 'BR', 106, 15, 205, FALSE, 'click', 'skip', FALSE, TRUE, 35000, FALSE, NULL, FALSE),
(4, '2026-02-21 18:34:00', 'tv', 'AR', 107, 16, 206, FALSE, 'autoplay', 'endplay', TRUE, FALSE, 210000, FALSE, NULL, FALSE),
(4, '2026-02-21 19:05:27', 'mobile', 'AR', 108, 16, 206, FALSE, 'click', 'endplay', FALSE, FALSE, 240000, TRUE, 1708654500, TRUE),
(5, '2026-02-20 14:10:12', 'desktop', 'BR', 109, 17, 207, TRUE, 'click', 'endplay', FALSE, FALSE, 600000, FALSE, NULL, FALSE),
(5, '2026-02-20 14:50:45', 'mobile', 'BR', 110, 18, 208, FALSE, 'autoplay', 'skip', TRUE, TRUE, 42000, FALSE, NULL, FALSE);

INSERT INTO "Tracks" (id, uri, track_name) VALUES
(101, 'spotify:track:101', 'Faixa 101'),
(102, 'spotify:track:102', 'Faixa 102'),
(103, 'spotify:track:103', 'Faixa 103'),
(104, 'spotify:track:104', 'Faixa 104'),
(105, 'spotify:track:105', 'Faixa 105'),
(106, 'spotify:track:106', 'Faixa 106'),
(107, 'spotify:track:107', 'Faixa 107'),
(108, 'spotify:track:108', 'Faixa 108'),
(109, 'spotify:track:109', 'Faixa 109'),
(110, 'spotify:track:110', 'Faixa 110');

INSERT INTO "Artist" (id, uri, artist_name) VALUES
(11, 'spotify:artist:11', 'Artista 11'),
(12, 'spotify:artist:12', 'Artista 12'),
(13, 'spotify:artist:13', 'Artista 13'),
(14, 'spotify:artist:14', 'Artista 14'),
(15, 'spotify:artist:15', 'Artista 15'),
(16, 'spotify:artist:16', 'Artista 16'),
(17, 'spotify:artist:17', 'Artista 17'),
(18, 'spotify:artist:18', 'Artista 18');

INSERT INTO "Album" (id, uri, album_name) VALUES
(201, 'spotify:album:201', 'Album 201'),
(202, 'spotify:album:202', 'Album 202'),
(203, 'spotify:album:203', 'Album 203'),
(204, 'spotify:album:204', 'Album 204'),
(205, 'spotify:album:205', 'Album 205'),
(206, 'spotify:album:206', 'Album 206'),
(207, 'spotify:album:207', 'Album 207'),
(208, 'spotify:album:208', 'Album 208');

INSERT INTO "LibraryTracks" (
    user_id,
    track_id,
    artist_id,
    album_id,
    in_playlist,
    liked
) VALUES
(1, 101, 11, 201, TRUE, TRUE),
(1, 102, 12, 202, FALSE, TRUE),
(2, 103, 13, 203, TRUE, FALSE),
(2, 104, 13, 203, TRUE, TRUE),
(3, 105, 14, 204, FALSE, TRUE),
(3, 106, 15, 205, TRUE, TRUE),
(4, 107, 16, 206, FALSE, FALSE),
(4, 108, 16, 206, TRUE, TRUE),
(5, 109, 17, 207, FALSE, TRUE),
(5, 110, 18, 208, TRUE, FALSE);

INSERT INTO "SoundCapsule" (
    id,
    user_id,
    date,
    "streamCount",
    "secondsPlayed"
) VALUES
(1, 1, '2026-01-01', 25, 3600),
(2, 2, '2026-01-01', 18, 2700),
(3, 3, '2026-01-01', 32, 5000),
(4, 4, '2026-01-01', 12, 1800),
(5, 5, '2026-01-01', 27, 4200);

INSERT INTO "SoundCapsule_TopTracks" (
    soundcapsule_id,
    track_id,
    "streamCount"
) VALUES
(1, 101, 10),
(1, 102, 8),
(2, 103, 6),
(2, 104, 5),
(3, 105, 12),
(3, 106, 9),
(4, 107, 4),
(5, 109, 11),
(5, 110, 7);

INSERT INTO "SoundCapsule_TopArtists" (
    soundcapsule_id,
    artist_id,
    "streamCount"
) VALUES
(1, 11, 15),
(1, 12, 10),
(2, 13, 7),
(3, 14, 14),
(3, 15, 8),
(4, 16, 5),
(5, 17, 9),
(5, 18, 6);

INSERT INTO "SoundCapsule_TopGenres" (
    soundcapsule_id,
    genre_name
) VALUES
(1, 'rock'),
(1, 'pop'),
(2, 'indie'),
(3, 'metal'),
(3, 'alternativo'),
(4, 'trap'),
(5, 'eletronico'),
(5, 'synthwave');

INSERT INTO "SoundCapsule_TimeOfDayStats" (
    soundcapsule_id,
    period,
    "secondsPlayed"
) VALUES
(1, 'morning', 1200),
(1, 'evening', 2400),
(2, 'afternoon', 1500),
(3, 'night', 3000),
(4, 'evening', 900),
(5, 'morning', 1800),
(5, 'night', 2400);

