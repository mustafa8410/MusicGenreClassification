package com.example.MusicGenreClassification.repository;

import com.example.MusicGenreClassification.entity.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    List<Prediction> findByAudio_UserIdOrderByPredictionDateDesc(Long userId);
    List<Prediction> findByAudio_GuestIdOrderByPredictionDateDesc(String guestId);
}
