package com.example.MusicGenreClassification.repository;

import com.example.MusicGenreClassification.entity.Audio;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface AudioRepository  extends JpaRepository<Audio, Long> {
    List<Audio> findAllByUserId(Long userId);
}
