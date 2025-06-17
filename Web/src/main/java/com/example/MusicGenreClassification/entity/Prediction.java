package com.example.MusicGenreClassification.entity;

import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Table(name = "prediction")
@Data
@Entity
public class Prediction {

    @Id
    @GeneratedValue(strategy = jakarta.persistence.GenerationType.IDENTITY)
    private Long id;
    private String genre;
    private double confidence;
    private LocalDateTime predictionDate;

    @OneToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "audio_id", nullable = false)
    private Audio audio;
}
